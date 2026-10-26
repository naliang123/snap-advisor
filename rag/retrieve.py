"""检索：向量 / BM25 / 混合（RRF 融合）/ 可选重排。

四种策略都可以单独跑，为的是能用实验数据回答一个真问题：
    "语义向量真的比关键词检索好吗？重排值不值得那一次额外调用？"
答案写在 evals/rag_ablation.py 的对比表里，不靠直觉。
"""
import config
from rag import embed, store

# RRF（倒数排名融合）常数：越大则排名靠前与靠后的差距被压缩得越平
RRF_K = 60
# 每种策略先各自取候选，再融合/重排
CANDIDATE_K = 50


def search(query: str, top_k: int = 5, mode: str = None, rerank: bool = None):
    """统一检索入口。返回 [{chunk_id, doc_id, source, text, score}]。"""
    mode = (mode or config.RAG_MODE or "hybrid").lower()
    if rerank is None:
        rerank = config.RAG_RERANK

    if mode == "vector":
        hits = _vector(query, top_k)
    elif mode == "bm25":
        hits = _bm25(query, top_k)
    elif mode == "hybrid":
        hits = _hybrid(query, top_k)
    else:
        raise ValueError("未知检索模式：%s（可选 vector / bm25 / hybrid）" % mode)

    if rerank and hits:
        hits = _rerank(query, hits, top_k)
    return hits


def _vector(query: str, top_k: int):
    qv = embed.embed_one(query)
    if not qv:
        return []
    scored = []
    for cid, doc, src, text, vec in store.all_vectors(config.EMBED_MODE, len(qv)):
        s = embed.cosine(qv, vec)
        if s > 0:
            scored.append({"chunk_id": cid, "doc_id": doc, "source": src,
                           "text": text, "score": s})
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def _bm25(query: str, top_k: int):
    pairs = store.fts_search(query, CANDIDATE_K)
    if not pairs:
        return []
    texts = store.get_texts([cid for cid, _ in pairs])
    out = []
    for cid, score in pairs:
        row = texts.get(cid)
        if not row:
            continue
        out.append({"chunk_id": cid, "doc_id": row["doc_id"],
                    "source": row["source"], "text": row["text"], "score": score})
    return out[:top_k]


def _hybrid(query: str, top_k: int):
    """混合检索：向量与 BM25 各自排名，用 RRF 融合。

    为什么用 RRF 而不是把两个分数加权相加：余弦相似度（0~1）和 BM25（负数）
    量纲完全不同，强行加权需要归一化，且对分布敏感；RRF 只用排名，天然可比。
    """
    vec_hits = _vector(query, CANDIDATE_K)
    bm_hits = _bm25(query, CANDIDATE_K)
    if not vec_hits:
        return bm_hits[:top_k]
    if not bm_hits:
        return vec_hits[:top_k]

    scores, meta = {}, {}
    for rank, h in enumerate(vec_hits):
        scores[h["chunk_id"]] = scores.get(h["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        meta[h["chunk_id"]] = h
    for rank, h in enumerate(bm_hits):
        scores[h["chunk_id"]] = scores.get(h["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        meta.setdefault(h["chunk_id"], h)

    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    out = []
    for cid, s in fused[:top_k]:
        item = dict(meta[cid])
        item["score"] = s
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# 重排（rerank）：先粗召回一批，再精挑——是提升 Top-K 质量最划算的一步
# ---------------------------------------------------------------------------

RERANK_PROMPT = (
    "下面是检索到的候选片段，请判断每条与问题的相关程度，按 0~10 打分。\n"
    "只输出 JSON：{\"scores\": [{\"id\": 0, \"score\": 8}, ...]}，不要解释。\n\n"
    "问题：%s\n\n候选：\n%s"
)


def _rerank(query: str, hits: list, top_k: int):
    if config.LLM_MODE == "real":
        try:
            return _llm_rerank(query, hits, top_k)
        except Exception:  # noqa: BLE001  重排失败不该让检索整体失败
            pass
    return _lexical_rerank(query, hits, top_k)


def _llm_rerank(query: str, hits: list, top_k: int):
    from llm import client
    listing = "\n".join("[%d] %s" % (i, h["text"][:120]) for i, h in enumerate(hits[:20]))
    msg = client.chat([{"role": "user", "content": RERANK_PROMPT % (query, listing)}])
    order = _parse_scores(getattr(msg, "content", "") or "")
    if not order:
        return _lexical_rerank(query, hits, top_k)
    ranked = []
    for idx, sc in order:
        if 0 <= idx < len(hits):
            item = dict(hits[idx])
            item["score"] = float(sc)
            ranked.append(item)
    # 未覆盖到的按原顺序补在后面，保证不丢候选
    seen = {r["chunk_id"] for r in ranked}
    ranked.extend(h for h in hits if h["chunk_id"] not in seen)
    return ranked[:top_k]


def _parse_scores(text: str):
    import json
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except Exception:
        return []
    out = []
    for item in data.get("scores") or []:
        try:
            out.append((int(item["id"]), float(item["score"])))
        except Exception:
            continue
    out.sort(key=lambda x: -x[1])
    return out


def _lexical_rerank(query: str, hits: list, top_k: int):
    """离线重排：按"问题里的词在片段中出现的比例"打分（词面重合，非语义）。"""
    qt = set(embed._tokens(query))
    out = []
    for h in hits:
        ht = set(embed._tokens(h["text"]))
        cover = (len(qt & ht) / len(qt)) if qt else 0.0
        item = dict(h)
        item["score"] = round(cover, 4)
        out.append(item)
    out.sort(key=lambda x: -x["score"])
    return out[:top_k]
