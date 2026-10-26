"""LangGraph 节点实现与路由函数。

图结构：
    START → perceive → agent ⇄ tools（工具循环，受 MAX_TOOL_ROUNDS 限制）
                        ↓
                     confirm（HITL 门：草稿含购买建议时征询用户）
                        ↓ regenerate（用户拒绝建议 → 回到 agent 重新生成）
                     finalize → END

节点都是纯函数：输入 state，返回"增量更新字典"，由 LangGraph 合并。
"""
import json
import sys

import config
import evidence as evidence_mod
import logging_utils as telemetry
from agent import streaming
from llm import client as llm_client
from llm import vlm
from memory import store
from tools.registry import TOOLS, run_tool


# ---------------------------------------------------------------------------
# 节点：感知
# ---------------------------------------------------------------------------

def perceive_node(state: dict) -> dict:
    """VLM 读图 → 结构化感知（真实或离线 mock 标注）。

    感知失败不炸主流程：显式返回"空观察 + 失败原因"，
    下游回答会如实告诉用户"没识别出画面信息"——这也是证据契约的一部分。
    """
    em = streaming.current()
    if em:
        em.emit("phase", stage="perceive", label="正在看这张图…")
    try:
        p = vlm.perceive(state.get("image_path"))
    except Exception as e:  # noqa: BLE001
        p = {
            "scene": "感知失败",
            "items": [], "read_text": [],
            "uncertain": ["视觉感知出错：%s" % str(e)[:120]],
            "evidence": {"source": "error", "note": str(e)[:160]},
        }
    telemetry.event("perceive", {
        "source": (p or {}).get("evidence", {}).get("source"),
        "items": len((p or {}).get("items") or []),
    })
    if em:
        em.emit("perception", data=p)
    return {"perception": p}


# ---------------------------------------------------------------------------
# 节点：对话决策（function calling）
# ---------------------------------------------------------------------------

def agent_node(state: dict) -> dict:
    """调用对话模型：决定发起工具调用，或直接给出回答。"""
    loose = bool(state.get("loose", not config.EVIDENCE_STRICT))

    system = _build_system(state, loose)
    history = [m for m in state.get("messages", []) if m.get("role") != "system"]
    messages = [{"role": "system", "content": system}] + history

    # 防死循环：达到工具轮数上限后不再下发工具，模型只能直接作答
    allow_tools = state.get("tool_rounds", 0) < config.MAX_TOOL_ROUNDS
    tools = TOOLS if allow_tools else None

    # 流式路径：边生成边推送增量；未绑定发射器时与原来完全一致（一次性）
    em = streaming.current()
    if em:
        em.emit("phase", stage="think", label="正在组织回答…")
        msg = llm_client.chat_stream(
            messages,
            tools=tools,
            on_delta=lambda t: em.emit("delta", text=t),
        )
        # 该轮最终发起了工具调用 → 刚才吐出的文字属于"边想边说"，撤回
        if getattr(msg, "tool_calls", None):
            em.emit("rollback")
    else:
        msg = llm_client.chat(messages, tools=tools)

    d = _message_to_dict(msg)
    telemetry.event("agent", {
        "tool_calls": [tc["function"]["name"] for tc in (d.get("tool_calls") or [])],
        "allow_tools": allow_tools,
    })
    return {"messages": state.get("messages", []) + [d]}


def route_after_agent(state: dict) -> str:
    """路由：assistant 发起了工具调用 → tools；否则 → confirm。"""
    messages = state.get("messages") or [{}]
    if (messages[-1] or {}).get("tool_calls"):
        return "tools"
    return "confirm"


# ---------------------------------------------------------------------------
# 节点：工具执行
# ---------------------------------------------------------------------------

def tools_node(state: dict) -> dict:
    """执行 assistant 发起的全部工具调用，结果写回消息与 tool_results。"""
    last = state["messages"][-1]
    new_messages, results = [], list(state.get("tool_results", []))
    em = streaming.current()
    for tc in last.get("tool_calls", []):
        name = tc["function"]["name"]
        if em:
            em.emit("phase", stage="tool", name=name,
                    label=("正在比价…" if name == "compare_price" else "正在查询…"))
        out = run_tool(name, tc["function"]["arguments"])
        results.append({"tool": name, "output": out})
        new_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(out, ensure_ascii=False),
        })
        telemetry.event("tool", {"name": name, "evidence": out.get("evidence")})
        if em:
            em.emit("tool", name=name, output=out)
            # 链接尽早推送：用户不必等正文生成完就能看到候选商品页
            links = _collect_links(results)
            if links:
                sig = "|".join(sorted(x["url"] for x in links))
                if sig != em.last_links_sig:   # 内容没变就不重复推
                    em.last_links_sig = sig
                    em.emit("links", items=links)
    return {
        "messages": state["messages"] + new_messages,
        "tool_results": results,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
    }


# ---------------------------------------------------------------------------
# 节点：HITL 人工确认门
# ---------------------------------------------------------------------------

_ADVICE_MARKERS = ("建议买", "建议不买", "值得买", "不建议", "直接买", "别买")


def confirm_node(state: dict) -> dict:
    """HITL：草稿含购买建议且开启 SNAP_CONFIRM 时，先征询用户意见。

    - 用户接受 → 直接进入 finalize；
    - 用户拒绝 → 注入"只要事实"的指令，回到 agent 重新生成。
    Web 模式（非交互终端）自动放行，避免阻塞请求。
    """
    last = _last_assistant(state)
    draft = (last or {}).get("content") or ""
    interactive = sys.stdin is not None and sys.stdin.isatty()

    need_confirm = (
        config.CONFIRM_MODE
        and not state.get("confirmed")
        and interactive
        and any(k in draft for k in _ADVICE_MARKERS)
    )
    if not need_confirm:
        return {"confirmed": True}

    print("\n[草稿建议] " + draft)
    ans = input("是否接受以上建议作为最终输出？(y=接受 / n=改为仅陈述事实) ").strip().lower()
    if ans == "n":
        return {
            "confirmed": True,
            "regenerate": True,
            "messages": state["messages"] + [{
                "role": "user",
                "content": "我暂时不想听购买建议，请只陈述画面与工具结果里的事实。",
            }],
        }
    return {"confirmed": True}


def route_after_confirm(state: dict) -> str:
    return "agent" if state.get("regenerate") else "finalize"


# ---------------------------------------------------------------------------
# 节点：收尾（证据校验 + 落库）
# ---------------------------------------------------------------------------

LINK_BLOCK_HEAD = "【参考链接】"


def finalize_node(state: dict) -> dict:
    """取最后一条 assistant 回答作为最终答案；做证据校验并写入对话存档。

    候选链接由本节点【确定性补全】：无论模型有没有主动写出链接，检索到的
    每条商品页链接都会附在回答末尾（去重、带价格、标注不确定性）——
    做到"说了更便宜就必须给链接"。证据校验只针对模型自己说的话，
    链接块里的价格来自工具结果，因此不会产生新的违规。
    """
    last = _last_assistant(state)
    answer = (last or {}).get("content") or "（无回答）"

    links = _collect_links(state.get("tool_results", []))
    checked = evidence_mod.check_answer(answer, state)
    full = _append_links(answer, links)

    store.log_turn(
        state.get("question", ""), full, state.get("image_path"),
        json.dumps(checked["violations"], ensure_ascii=False),
    )
    telemetry.event("finalize", {
        "violations": checked["violations"], "links": len(links),
    })
    em = streaming.current()
    if em:
        em.emit("violations", items=checked["violations"])
    return {"final_answer": full, "violations": checked["violations"], "links": links}


def _collect_links(tool_results) -> list:
    """收集候选商品链接（去重，保留平台/标题/价格）。

    不判断是否"最低价"——诚实原则是：把检索到的链接全部列出并说明不确定。
    """
    items, seen = [], set()
    for tr in tool_results or []:
        if tr.get("tool") != "compare_price":
            continue
        for r in ((tr.get("output") or {}).get("results") or []):
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            items.append({
                "platform": r.get("platform") or "网页",
                "title": (r.get("title") or "")[:30],
                "price": r.get("price"),
                "url": url,
            })
    return items


def _append_links(answer: str, links: list) -> str:
    """把候选链接确定性地补到回答末尾（流式与一次性两条路径共用，结果一致）。"""
    if not links:
        return answer
    lines = []
    for it in links:
        price = ("￥%s " % it["price"]) if it.get("price") else "价格未取到 "
        lines.append("- %s %s%s\n  %s" % (it["platform"], price, it.get("title") or "",
                                          it["url"]))
    missing = sum(1 for x in links if not x.get("price"))
    if missing == len(links):
        tail = "（这些页面都没直接显示价格，是否最低价无法确认，供你自行比照。）"
    elif missing:
        tail = "（部分页面没取到价格；链接按检索顺序列出，是否最低价无法保证。）"
    else:
        tail = "（链接按检索顺序列出，供你自行核对；是否最低价无法保证。）"
    return "%s\n\n%s\n%s\n%s" % (answer.rstrip(), LINK_BLOCK_HEAD,
                                 "\n".join(lines), tail)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_system(state: dict, loose: bool) -> str:
    from agent import prompts
    return prompts.build_system(
        state.get("perception"), state.get("profile_facts"), loose
    )


def _last_assistant(state: dict):
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return m
    return None


def _message_to_dict(m) -> dict:
    """把 OpenAI message 对象（或 mock 同形对象）转成消息字典。"""
    d = {"role": "assistant", "content": getattr(m, "content", None)}
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return d
