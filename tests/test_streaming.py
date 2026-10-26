"""流式输出相关测试。

核心保证三条：
1. 流式路径与一次性路径的 final_answer【逐字相同】（三处逻辑不漂移）；
2. 逐块推送的 delta 拼接起来 == 最终答案，不多不少；
3. 检索到链接时必须直出：链接块与 links 字段都带上全部候选链接。
"""
import config
import agent.streaming as streaming
from agent import nodes


def _drain(emitter):
    """把队列里的事件全部取出（含哨兵前的全部事件）。"""
    out = []
    while True:
        item = emitter.queue.get_nowait()
        if item == streaming.Emitter.END:
            break
        out.append(item)
    return out


def _sample_path():
    """示例货架图（保证感知里有商品，才会走到工具调用路径）。"""
    from agent import runner
    runner.ensure_sample()
    return str(config.SAMPLE_IMAGE)


def _run_with_emitter(question, image_path):
    emitter = streaming.Emitter()
    token = streaming.bind(emitter)
    try:
        from agent import runner
        result = runner.run_once(question, image_path)
    finally:
        emitter.end()
        streaming.unbind(token)
    return result, _drain(emitter)


def test_stream_equals_once_result():
    """同一问题：流式与一次性两条路径的 final_answer 必须完全一致。"""
    q = "货架上的保温杯值不值得买？顺便帮我看看有没有更便宜的。"
    image = _sample_path()
    result_stream, events = _run_with_emitter(q, image)
    from agent import runner
    result_once = runner.run_once(q, image)

    assert result_stream["final_answer"] == result_once["final_answer"]
    assert result_stream["violations"] == result_once["violations"]
    assert result_stream["perception"] == result_once["perception"]

    # 节点层应吐出进度、感知、工具结果、增量与校验结果
    types = [e["type"] for e in events]
    for expected in ("phase", "perception", "tool", "delta", "violations"):
        assert expected in types


def test_deltas_join_to_answer():
    """逐块推送的内容拼接起来必须等于最终答案（不多不少）。"""
    result, events = _run_with_emitter("货架上的保温杯值不值？帮我比价",
                                       _sample_path())
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas, "流式路径应至少吐出一块内容"
    assert "".join(deltas) == result["final_answer"]


def test_rollback_before_deltas_on_tool_turn():
    """模型那一轮最终发起了工具调用时，必须先撤回本轮可能吐出的文字。"""
    result, events = _run_with_emitter("这个保温杯多少钱？帮我比价",
                                       _sample_path())
    types = [e["type"] for e in events]
    assert "tool" in types, "比价问题应触发工具调用"
    if "rollback" in types:
        # 撤回事件必须早于正文增量，正文里不能混入"边想边说"的内容
        assert types.index("rollback") < types.index("delta")
    # 正文内容仍与一次性结果一致
    from agent import runner
    assert result["final_answer"] == runner.run_once(
        "这个保温杯多少钱？帮我比价", _sample_path())["final_answer"]


def test_collect_links_dedup_and_only_price_tool():
    tool_results = [
        {"tool": "compare_price", "output": {"results": [
            {"platform": "www.jd.com", "title": "A 商品", "price": "89.25",
             "url": "https://item.jd.com/1.html"},
            {"platform": "www.jd.com", "title": "A 商品（重复）", "price": "89.25",
             "url": "https://item.jd.com/1.html"},
            {"platform": "www.taobao.com", "title": "B 商品", "price": None,
             "url": "https://www.taobao.com/2.htm"},
        ]}},
        {"tool": "query_profile", "output": {"facts": [
            {"fact": "预算不超过50", "url": "https://should-not-appear"}]}},
    ]
    links = nodes._collect_links(tool_results)
    assert len(links) == 2                     # 去重 + 只取比价工具
    assert links[0]["url"] == "https://item.jd.com/1.html"
    assert links[1]["price"] is None
    assert all(l["url"] != "https://should-not-appear" for l in links)


def test_append_links_includes_urls_and_disclaimer():
    links = [
        {"platform": "www.jd.com", "title": "A", "price": "89.25",
         "url": "https://item.jd.com/1.html"},
        {"platform": "www.taobao.com", "title": "B", "price": None,
         "url": "https://www.taobao.com/2.htm"},
    ]
    answer = "画面价 ￥99，网上有更便宜的。"
    full = nodes._append_links(answer, links)
    assert "https://item.jd.com/1.html" in full     # 低价链接直接给出
    assert "https://www.taobao.com/2.htm" in full   # 不确定的也全部列出
    assert "价格未取到" in full
    assert "是否最低价无法保证" in full             # 明确不确定性
    assert full.startswith(answer)


def test_append_links_noop_without_links():
    answer = "画面里只有这一个价签。"
    assert nodes._append_links(answer, []) == answer
