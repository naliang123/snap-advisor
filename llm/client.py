"""对话模型客户端：真实（OpenAI 兼容）与离线 Mock 双模式。

Mock 大脑是一套【确定性规则】：从消息历史里解析用户意图 / 工具结果 /
系统提示中的画面感知 JSON，决定发起工具调用还是直接作答。它保证：
  1) 零 API Key 也能完整跑通 Agent 循环（含 function calling）；
  2) 评测与消融实验可复现——同样的输入永远得到同样的输出。

Mock 的"loose 模式"（消融实验用）会故意编造无依据价格（¥99 / ¥9.9），
用于演示"关闭证据约束后无证据断言率上升"，会被 evidence.py 检出。
"""
import json
import re
import types

import config

_real_client = None  # 惰性缓存真实客户端


def get_chat_client():
    if config.LLM_MODE == "mock":
        return _MockClient()
    global _real_client
    if _real_client is None:
        from openai import OpenAI
        _real_client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL
        )
    return _real_client


def chat(messages, tools=None, temperature=0.3, max_tokens=800):
    """调用对话模型，返回 assistant message（含 .content / .tool_calls）。"""
    client = get_chat_client()
    kwargs = dict(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
    resp = _create_compat(client, kwargs)
    return resp.choices[0].message


def _create_compat(client, kwargs):
    """供应商兼容层：部分模型（如 kimi-k3）不接受 temperature 参数，
    遇到此类 400 错误时自动去掉该参数重试一次。"""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        if "temperature" in str(e) and "temperature" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
            return client.chat.completions.create(**kwargs)
        raise


# ---------------------------------------------------------------------------
# 流式调用：逐块回调 on_delta(text)，返回与 chat() 同形的 message 对象
# ---------------------------------------------------------------------------

# mock 模式的"流式"节奏：每块字符数与间隔（秒），仅用于让离线演示也有一致的观感
MOCK_CHUNK_CHARS = 24
MOCK_CHUNK_INTERVAL = 0.025


def chat_stream(messages, tools=None, on_delta=None, temperature=0.3, max_tokens=800):
    """流式调用对话模型。

    与 chat() 的差异只有"边生成边回调"，最终返回的 message 与一次性调用
    同形（.content / .tool_calls），因此流式与一次性结果完全一致：
    - 真实模式：直接透传模型原生 chunk（不人为限速，最小延迟）；
    - mock 模式：把已有答案按 24 字符/块、25ms 间隔切分播放。
    """
    if config.LLM_MODE == "mock":
        msg = _mock_reply(messages, tools)
        if msg.content:
            _emit_in_blocks(msg.content, on_delta)
        return msg

    client = get_chat_client()
    kwargs = dict(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    if tools:
        kwargs["tools"] = tools
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        if "temperature" in str(e):
            kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise

    parts, tool_parts = [], {}
    for chunk in resp:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            parts.append(text)
            if on_delta:
                on_delta(text)
        # 工具调用是分片到达的（name 与 arguments 分多次），按 index 组装
        for tc in (getattr(delta, "tool_calls", None) or []):
            slot = tool_parts.setdefault(
                tc.index,
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn:
                if getattr(fn, "name", None):
                    slot["function"]["name"] += fn.name
                if getattr(fn, "arguments", None):
                    slot["function"]["arguments"] += fn.arguments

    content = "".join(parts) or None
    tool_calls = None
    if tool_parts:
        tool_calls = [
            _Msg(id=slot["id"] or ("call_%s" % idx), type="function",
                 function=_Msg(name=slot["function"]["name"],
                               arguments=slot["function"]["arguments"] or "{}"))
            for idx, slot in sorted(tool_parts.items())
        ]
    return _Msg(content=content, tool_calls=tool_calls)


def _emit_in_blocks(text: str, on_delta, size: int = MOCK_CHUNK_CHARS,
                    interval: float = MOCK_CHUNK_INTERVAL):
    """把整段文本切成小块逐步回调（mock / 无原生流式的供应商兜底）。"""
    import time
    if not on_delta:
        return
    for i in range(0, len(text), size):
        on_delta(text[i:i + size])
        time.sleep(interval)



# ---------------------------------------------------------------------------
# 离线 Mock 大脑（仅 LLM_MODE=mock 时启用）
# ---------------------------------------------------------------------------

def _parse_block(system_text: str, tag: str):
    """从系统提示中提取【tag】...【/tag】包裹的 JSON 块。"""
    m = re.search(r"【%s】(.*?)【/%s】" % (tag, tag), system_text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except Exception:
        return None


def _pick_item(perception: dict, text: str):
    """优先选择用户问题里点名的商品；否则取第一个。"""
    items = (perception or {}).get("items") or []
    if not items:
        return None
    for it in items:
        if it.get("name") and it["name"] in (text or ""):
            return it
    return items[0]


def _items_text(perception: dict) -> str:
    items = (perception or {}).get("items") or []
    lines = []
    for it in items:
        price = it.get("price")
        price_s = " ￥%s" % price if price else "（没看到价签）"
        promo = ("，促销：%s" % it["promo"]) if it.get("promo") else ""
        conf = it.get("confidence")
        unsure = "（拿不准）" if isinstance(conf, (int, float)) and conf < 0.5 else ""
        lines.append("- %s%s%s%s" % (it.get("name", "未知商品"), price_s, promo, unsure))
    return "\n".join(lines) if lines else "- 画面里没有识别到商品"


def _tool_call_msg(name: str, arguments: dict, call_id: str = "call_mock_1"):
    """构造带 function calling 的 assistant message（OpenAI 响应形状）。"""
    return _Msg(
        content=None,
        tool_calls=[_Msg(
            id=call_id, type="function",
            function=_Msg(name=name,
                          arguments=json.dumps(arguments, ensure_ascii=False)),
        )],
    )


class _Msg(types.SimpleNamespace):
    """与 OpenAI 返回的 message 对象同形（content / tool_calls）。"""


class _MockCompletions:
    def create(self, *, messages, tools=None, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=_mock_reply(messages, tools))]
        )


class _MockClient:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_MockCompletions())


def _mock_reply(messages, tools=None):
    system = next((m["content"] for m in messages if m.get("role") == "system"), "") or ""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    last_user = (user_msgs[-1]["content"] if user_msgs else "") or ""
    perception = _parse_block(system, "画面感知")
    loose = "证据约束已关闭" in system
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    has_tools = bool(tools)

    # 1) 工具结果刚返回 → 生成最终回答
    if messages and messages[-1].get("role") == "tool":
        try:
            result = json.loads(tool_msgs[-1]["content"])
        except Exception:
            result = {}
        return _Msg(content=_answer_after_tool(result, perception, loose), tool_calls=None)

    text = last_user

    # 2) 回忆类意图 → search_kb（放在比价之前：问"上次那款多少钱"是在回忆，
    #    不是在查当前商品的线上价，两者不能混淆）
    if has_tools and any(k in text for k in
                         ("上次", "以前", "之前", "那款", "说过", "提过", "还记得")):
        return _tool_call_msg("search_kb", {"query": text.strip()[:60], "top_k": 5})

    # 3) 比价意图 → 发起 compare_price 工具调用
    if has_tools and any(k in text for k in ("比价", "便宜", "多少钱", "贵不贵", "线上")):
        item = _pick_item(perception, text)
        if item:
            return _tool_call_msg("compare_price", {"query": item.get("name") or "商品"})

    # 3) 记偏好意图 → remember_preference
    m = re.search(r"记住[：:]?\s*(.*)", text)
    if has_tools and (m or "我喜欢" in text or "预算" in text):
        fact = (m.group(1).strip() if m else text.strip())[:80]
        return _tool_call_msg("remember_preference", {"fact": fact})

    # 4) 查画像意图
    if has_tools and any(k in text for k in ("我的偏好", "我之前", "我的画像", "我记得")):
        return _tool_call_msg("query_profile", {})

    # 5) 直接作答（仅依据画面感知）
    return _Msg(content=_direct_answer(perception, loose, text), tool_calls=None)


def _answer_after_tool(result: dict, perception: dict, loose: bool) -> str:
    # 知识库/记忆检索的结果（与比价结果同形，靠 evidence.source 区分）
    if (result.get("evidence") or {}).get("source") == "knowledge_base":
        items = result.get("results") or []
        if not items:
            return ("我在自己的记忆和知识库里检索过，没有找到相关内容。"
                    "我不凭印象编——你可以再说清楚一点，我记下来以后就能检索到。")
        lines = ["我检索到这些相关记录："]
        for r in items:
            lines.append("- %s [记忆:%s]（来自%s）" % (
                (r.get("text") or "").replace("\n", " ")[:80],
                r.get("chunk_id", ""), r.get("platform", "知识库")))
        lines.append("以上只来自检索到的记录，编号可溯源；没检索到的我不瞎说。")
        return "\n".join(lines)

    # compare_price 的结果
    if "results" in result:
        lines = ["我查到的线上报价："]
        for r in result["results"]:
            note = "（演示数据）" if "演示" in (r.get("note") or "") else ""
            lines.append("- %s：%s ￥%s%s" % (r.get("platform", "?"),
                                             r.get("title", ""), r.get("price", "?"), note))
        body = "\n".join(lines)
        demo = bool((result.get("evidence") or {}).get("is_demo"))
        item = _pick_item(perception, "")
        advice = ""
        if item and item.get("price"):
            advice = "画面价签是 ￥%s，可以和线上报价对比后再决定值不值。" % item["price"]
        if loose:
            # 消融模式：故意编造行情价格 ￥99（无任何依据），会被证据校验检出
            return body + " " + advice + "依我看这款网上行情一般要 ￥99 起，画面里这个价很划算，建议直接买！"
        tail = "注意：以上为演示数据，配置真实比价数据源后才会给出真实报价。" if demo else ""
        return (body + " " + advice + " " + tail).strip()

    # remember_preference 的结果
    if result.get("status") == "已记住":
        return "好的，我记住了：%s。之后给建议时会参考这条偏好。" % result.get("fact", "")

    # query_profile 的结果
    if "facts" in result:
        facts = result.get("facts") or []
        if not facts:
            return "我目前还没记住你的任何偏好。"
        return "我记得你的偏好：\n" + "\n".join("- %s" % f for f in facts)

    return "（离线演示）工具已执行，但没有返回可用结果。"


def _direct_answer(perception: dict, loose: bool, question: str) -> str:
    if not perception or not perception.get("items"):
        return "这张图里我没识别出商品信息。可以拍清楚一点再发我，或直接告诉我商品名。"
    head = "从画面看：\n" + _items_text(perception)
    if loose:
        # 消融模式：故意编造无依据行情价格 ￥9.9
        return head + " 顺便说一句，网上同类一般 ￥9.9 就能买到，别买贵的！"
    tail = "以上只包含画面里明确可见的信息，没看到的属性我不猜。"
    if any(k in question for k in ("值", "买", "划算")):
        tail += "值不值可以结合价签和你的预算判断，需要线上比价可以让我查。"
    return head + " " + tail
