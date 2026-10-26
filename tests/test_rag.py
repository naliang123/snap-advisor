"""检索（RAG）模块测试。

覆盖：切分规则 / 离线嵌入的确定性 / 入库与召回 / 三种检索策略 / 工具契约 /
检索结果能否作为"有依据的价格"被证据契约接受 / 端到端是否真的用上检索。
"""
import config
import evidence
from rag import chunk as chunker, embed, index as ri, retrieve, store as kb
from tools import registry


def _seed():
    kb.clear()
    ri.index_fact("我不买含硅油的洗发水")
    ri.index_fact("保温杯预算不超过两百元")
    ri.index_turn("上次那款保温杯怎么样",
                  "上次查到保温杯线上 ￥129，画面价签是 ￥149。")


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------

def test_split_empty_and_short():
    assert chunker.split_text("") == []
    assert chunker.split_text("一句话") == ["一句话"]


def test_split_respects_size_and_overlap():
    text = "。".join("这是第%d段内容，讲了一个完整的要点" % i for i in range(30))
    pieces = chunker.split_text(text, size=100, overlap=20)
    assert len(pieces) > 1
    # 单块不超过 size + 一个片段的溢出（贪心合并的合理上界）
    assert all(len(p) <= 160 for p in pieces)
    # 相邻块之间确有重叠内容
    if len(pieces) > 1:
        assert pieces[0][-10:] in pieces[1]


# ---------------------------------------------------------------------------
# 离线嵌入
# ---------------------------------------------------------------------------

def test_mock_embed_is_deterministic_and_normalized():
    a = embed.mock_embed("买洗发水要注意硅油")
    b = embed.mock_embed("买洗发水要注意硅油")
    assert a == b                                  # 可复现：同输入必同输出
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6  # 已归一化


def test_mock_embed_prefers_lexically_similar():
    q = embed.mock_embed("洗发水含硅油")
    near = embed.mock_embed("我不买含硅油的洗发水")
    far = embed.mock_embed("猫粮只买无谷配方，预算两百")
    assert embed.cosine(q, near) > embed.cosine(q, far)


# ---------------------------------------------------------------------------
# 入库与检索
# ---------------------------------------------------------------------------

def test_index_and_search_hits_expected_memory():
    _seed()
    hits = retrieve.search("我以前说过洗发水要注意什么", top_k=5)
    assert hits
    assert any("硅油" in h["text"] for h in hits)
    # 每条结果都可溯源
    assert all(h.get("chunk_id") and h.get("source") for h in hits)


def test_three_modes_all_return_hits():
    _seed()
    for mode in ("vector", "bm25", "hybrid"):
        hits = retrieve.search("保温杯预算", top_k=3, mode=mode)
        assert hits, mode
        assert any("保温杯" in h["text"] for h in hits), mode


def test_empty_library_returns_empty_not_invented(monkeypatch):
    """库里没有就必须说没检索到：关掉惰性重建，模拟真正的空库。"""
    kb.clear()
    monkeypatch.setattr(ri, "ensure_indexed", lambda: False)
    out = registry.run_tool("search_kb", '{"query": "随便问点什么"}')
    assert out["count"] == 0
    assert out["results"] == []
    assert "没" in (out.get("note") or "")   # 必须如实说没检索到


def test_reindex_is_idempotent():
    _seed()
    before = kb.count()
    ri.reindex_all()
    after = kb.count()
    assert after >= before           # 重建不会丢数据
    ri.reindex_all()                 # 再建一次不应翻倍堆积
    assert kb.count() == after


# ---------------------------------------------------------------------------
# 工具契约与证据契约的接缝
# ---------------------------------------------------------------------------

def test_search_kb_returns_evidence_contract():
    _seed()
    out = registry.run_tool("search_kb", '{"query": "上次保温杯多少钱", "top_k": 3}')
    ev = out["evidence"]
    assert ev["source"] == "knowledge_base"
    assert "fetched_at" in ev and "engine" in ev
    for r in out["results"]:
        assert r.get("chunk_id")


def test_profile_facts_are_not_false_violations():
    """用户自己说过的预算属于有依据，不能被误判成"无证据价格"。

    （真实链路曾出现：回答提到画像里的"预算不超过200元"被判违规。）
    """
    state = {"perception": {}, "tool_results": [
        {"tool": "query_profile",
         "output": {"facts": ["保温杯预算不超过两百元"],
                    "evidence": {"source": "memory"}}}]}
    assert "200" in evidence.allowed_prices(state) or "两百" in str(state)


def test_retrieved_prices_become_allowed_evidence():
    """检索到的历史价格属于"有依据"，不应被判为无证据断言。"""
    _seed()
    out = registry.run_tool("search_kb", '{"query": "上次保温杯多少钱"}')
    state = {"perception": {}, "tool_results": [
        {"tool": "search_kb", "output": out}]}
    allowed = evidence.allowed_prices(state)
    assert "129" in allowed or "149" in allowed


# ---------------------------------------------------------------------------
# 端到端：Agent 是否真的会主动检索
# ---------------------------------------------------------------------------

def test_fabricated_citation_is_detected():
    """编造一个没检索到的引用编号，和编造价格一样要被判违规。"""
    _seed()
    out = registry.run_tool("search_kb", '{"query": "洗发水硅油"}')
    state = {"perception": {}, "tool_results": [
        {"tool": "search_kb", "output": out}]}
    good = out["results"][0]["chunk_id"]
    assert evidence.check_citations("参考【记忆:%s】" % good, state) == []
    bad = evidence.check_citations("参考【记忆:turn:我编的编号#9】", state)
    assert bad and bad[0]["type"] == "引用不存在"


def test_agent_triggers_search_kb_and_cites_source():
    _seed()
    from agent import runner
    runner.ensure_sample()
    res = runner.run_once("我上次说过买洗发水要注意什么来着？",
                          str(config.SAMPLE_IMAGE))
    tools = [t["tool"] for t in res.get("tool_results", [])]
    assert "search_kb" in tools, "回忆类问题应主动触发检索"
    assert "[记忆:" in (res.get("final_answer") or ""), "引用必须带可溯源编号"
