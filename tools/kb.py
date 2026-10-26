"""知识库/记忆检索工具：把"相关的旧信息"作为证据交给模型。

与比价工具同构（返回值一律带 evidence、results 带可溯源 id），区别只是
证据来源：比价来自外部世界，这里来自用户自己的记忆与本地知识库。

检索到的内容仍然只是【候选证据】，不是答案——模型能不能引用、会不会被判
无证据断言，仍旧由 evidence.py 说了算。这一点是本模块与主线的接缝。
"""
import time

import config
from rag import index as rag_index, retrieve

_SOURCE_LABEL = {
    "memory_fact": "你的偏好",
    "memory_turn": "过去的问答",
    "kb_file": "本地知识库",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def search_kb(query: str, top_k: int = 5) -> dict:
    query = (query or "").strip()
    if not query:
        return {"query": query, "count": 0, "results": [],
                "note": "没有给出检索问题",
                "evidence": {"source": "knowledge_base", "fetched_at": _now()}}

    try:
        rag_index.ensure_indexed()
        hits = retrieve.search(query, top_k=top_k)
    except Exception as e:  # noqa: BLE001  检索失败必须如实告知，不能假装查过
        return {"query": query, "count": 0, "results": [],
                "note": "检索失败：%s" % str(e)[:120],
                "evidence": {"source": "knowledge_base", "fetched_at": _now(),
                             "error": str(e)[:120]}}

    results = []
    for h in hits:
        results.append({
            "platform": _SOURCE_LABEL.get(h["source"], "知识库"),
            "title": h["doc_id"],
            "text": h["text"][:400],
            "score": round(float(h["score"]), 4),
            "chunk_id": h["chunk_id"],
        })

    out = {
        "query": query,
        "mode": "%s%s" % (config.RAG_MODE, "+rerank" if config.RAG_RERANK else ""),
        "count": len(results),
        "results": results,
        "evidence": {
            "source": "knowledge_base",
            "fetched_at": _now(),
            "is_demo": False,
            "engine": config.RAG_MODE,
            "embed_mode": config.EMBED_MODE,
        },
    }
    if not results:
        out["note"] = "库里没有检索到相关内容（诚实说没查到，不要凭印象回答）"
    elif config.EMBED_MODE == "mock":
        out["note"] = "离线词面检索（非语义）：按字面重合度召回"
    return out
