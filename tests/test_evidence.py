"""证据契约测试：无依据价格必须被检出。"""
import evidence


def _state(perception=None, tool_results=None):
    return {"perception": perception or {}, "tool_results": tool_results or []}


def test_fabricated_price_flagged():
    state = _state(perception={"items": [{"name": "保温杯", "price": "49.90"}],
                               "read_text": []})
    out = evidence.check_answer("依我看这款网上一般要 ¥99 起，建议买！", state)
    types = [v["type"] for v in out["violations"]]
    assert "无证据价格" in types, "编造的 ¥99 必须被检出"


def test_grounded_price_passes():
    state = _state(perception={"items": [{"name": "保温杯", "price": "49.90"}],
                               "read_text": []})
    out = evidence.check_answer("画面价签是 ¥49.90，没看到其他价签。", state)
    assert out["violations"] == []


def test_price_format_normalization():
    """'¥49.90' 与感知 '49.9' 视为同一价格（格式归一化）。"""
    state = _state(perception={"items": [{"name": "保温杯", "price": "49.9"}],
                               "read_text": []})
    out = evidence.check_answer("价签写着 ¥49.90。", state)
    assert out["violations"] == []


def test_tool_price_is_allowed():
    tool_results = [{"tool": "compare_price", "output": {
        "results": [{"platform": "京东", "price": "45.00"}],
        "evidence": {"source": "mock", "is_demo": True},
    }}]
    state = _state(perception={"items": [], "read_text": []},
                   tool_results=tool_results)
    out = evidence.check_answer("京东报价 ¥45.00（演示数据）。", state)
    assert out["violations"] == []


def test_demo_data_must_be_declared():
    tool_results = [{"tool": "compare_price", "output": {
        "results": [{"platform": "京东", "price": "45.00"}],
        "evidence": {"source": "mock", "is_demo": True},
    }}]
    state = _state(tool_results=tool_results)
    out = evidence.check_answer("京东报价 ¥45.00，很便宜！买！", state)
    types = [v["type"] for v in out["violations"]]
    assert "未声明演示数据" in types, "使用演示数据却未声明，必须被检出"


def test_yuan_suffix_extraction():
    assert evidence.extract_prices("这杯 12.5 元") == ["12.5"]
    assert evidence.extract_prices("促销价¥9.9") == ["9.9"]
    assert evidence.extract_prices("没有价格") == []


def test_derived_price_from_promotion_passes():
    """"第二件半价"推导出的合计/均摊属于"有依据的计算"，不该判违规。

    真实链路曾误判：画面 49.90 + 促销"第二件半价"，回答给出
    "两只共 74.85、折合 37.4 一只"，被判 2 处无证据价格。
    """
    state = _state(perception={
        "items": [{"name": "保温杯", "price": "49.90"}],
        "read_text": ["促销:第二件半价"],
    })
    out = evidence.check_answer(
        "这款 ¥49.90，第二件半价，买两只一共 ¥74.85，折合约 ¥37.4 一只。", state)
    assert out["violations"] == [], "有依据的计算价格不应判违规"


def test_derivation_does_not_allow_arbitrary_numbers():
    """放宽推导不等于放水：凭空出现、无法由证据算出的价格仍要判违规。"""
    state = _state(perception={
        "items": [{"name": "保温杯", "price": "49.90"}],
        "read_text": [],
    })
    out = evidence.check_answer("这款 ¥49.90，但别家只要 ¥88.88。", state)
    bad = [v for v in out["violations"] if v["type"] == "无证据价格"]
    assert any(v["price"] == "88.88" for v in bad)
