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
