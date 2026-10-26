"""工具层测试：比价契约 / 记忆读写。"""
from memory import store
from tools import price
from tools.registry import run_tool


def test_mock_price_known_product():
    out = price.search_price_offers("保温杯")
    assert out["results"], "知名商品应有演示报价"
    ev = out["evidence"]
    assert ev["is_demo"] is True
    assert ev["source"] == "mock"
    for r in out["results"]:
        assert "演示" in (r.get("note") or ""), "mock 结果必须标注演示数据"


def test_mock_price_generic_product():
    out = price.search_price_offers("洗衣液")
    assert len(out["results"]) >= 3
    assert out["evidence"]["is_demo"] is True


def test_mock_price_empty_query():
    out = price.search_price_offers("")
    assert out["results"] == []
    assert out["evidence"]["reason"] == "empty_product"


def test_registry_compare_price():
    out = run_tool("compare_price", '{"query": "保温杯"}')
    assert "results" in out and out["results"]
    assert out["evidence"]["is_demo"] is True


def test_registry_remember_and_query_profile():
    out = run_tool("remember_preference", '{"fact": "预算不超过50"}')
    assert out["status"] == "已记住"
    prof = run_tool("query_profile", "{}")
    assert "预算不超过50" in prof["facts"]


def test_registry_unknown_tool():
    out = run_tool("no_such_tool", "{}")
    assert "error" in out


def test_store_roundtrip():
    store.add_fact("不喜欢香菜")
    facts = store.get_facts()
    assert any("香菜" in f["fact"] for f in facts)
