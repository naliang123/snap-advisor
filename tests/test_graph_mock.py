"""Agent 主图测试（离线 mock）：工具循环 / 证据校验 / 消融对比 / HITL 骨架。"""
import config
from agent import runner
from app.server import app as fastapi_app  # 顺带冒烟：Web 应用可导入


def test_server_importable():
    assert fastapi_app.title


def test_full_compare_flow_strict():
    runner.ensure_sample()
    result = runner.run_once(
        "保温杯有没有更便宜的？帮我比价", str(config.SAMPLE_IMAGE))
    assert result.get("final_answer")
    tools = [t["tool"] for t in result.get("tool_results", [])]
    assert "compare_price" in tools, "比价意图必须触发工具"
    assert result.get("violations") == [], "严格模式下不应有无证据断言"


def test_loose_mode_creates_violations():
    result = runner.run_once(
        "保温杯有没有更便宜的？帮我比价", str(config.SAMPLE_IMAGE), loose=True)
    violations = result.get("violations") or []
    types = [v["type"] for v in violations]
    assert "无证据价格" in types, "消融（松弛）模式必须产生可检出的编造价格"


def test_remember_flow():
    result = runner.run_once("记住我预算不超过50", None)
    tools = [t["tool"] for t in result.get("tool_results", [])]
    assert "remember_preference" in tools
    # mock 回答会把记住的事实原样复述回来
    assert "预算不超过50" in (result.get("final_answer") or "")


def test_direct_answer_grounds_prices():
    result = runner.run_once("货架上都有什么？", str(config.SAMPLE_IMAGE))
    answer = result.get("final_answer") or ""
    assert "保温杯" in answer
    assert result.get("violations") == []


def test_confirm_does_not_leak_regenerate():
    """回归保护：用户拒绝建议后 regenerate=True，重新生成后再进 confirm 必须复位。

    LangGraph 的状态是增量合并：上一轮留下的 regenerate=True 如果不显式覆盖，
    route_after_confirm 会每轮都返回 "agent"，形成 agent ⇄ confirm 死循环
    （最终撞到 recursion_limit 才报错）。
    """
    from agent import nodes

    out = nodes.confirm_node({
        "messages": [{"role": "assistant", "content": "建议买这个，性价比高"}],
        "confirmed": True,      # 已经确认过一次
        "regenerate": True,     # 上一轮"拒绝"留下的脏状态
    })
    assert out["confirmed"] is True
    assert out.get("regenerate") is False, "必须显式复位 regenerate"
    assert nodes.route_after_confirm({"regenerate": out.get("regenerate")}) == "finalize"


def test_multi_turn_reuses_perception(monkeypatch):
    """多轮对话同一张图：第二轮应复用感知，不再调用 VLM（省时省钱）。"""
    from llm import vlm

    counter = {"n": 0}
    original = vlm.perceive

    def counted(path):
        counter["n"] += 1
        return original(path)

    monkeypatch.setattr(vlm, "perceive", counted)
    runner.ensure_sample()

    first = runner.run_once("货架上有什么？", str(config.SAMPLE_IMAGE))
    assert counter["n"] == 1, "第一轮必须调用一次 VLM"

    runner.run_once("那第二个呢？", str(config.SAMPLE_IMAGE),
                    messages=first.get("messages"),
                    perception=first.get("perception"))
    assert counter["n"] == 1, "第二轮传入了感知，不应重复调用 VLM"
