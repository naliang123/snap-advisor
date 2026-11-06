"""检索库存储：SQLite（chunk 文本 + FTS5 全文索引）+ Milvus Lite（向量）。

职责划分：
- SQLite 是【权威】：chunk 文本、来源、嵌入模式/维度、全文索引（BM25）。
  计数、取原文、幂等删除都以它为准，重建索引也从它出发。
- Milvus Lite 是【向量索引】：只存 chunk_id + 向量 + 少量过滤字段。
  它可被重建——`main.py --index` 会从 SQLite 的文本重新嵌入并写回。

为什么用 Milvus Lite 而不是自建暴力余弦：
- 它是真正的向量数据库（ANN 索引），但以【嵌入式】方式运行：pip 安装、
  单目录存储、不需要 Docker 或独立服务，符合本项目"clone 下来就能跑"的定位；
- collection 按【嵌入模式 + 维度】命名（如 kb_real_1024），换嵌入模型不会串向量。

中文全文检索的一个坑（SQLite FTS5）：
    unicode61 分词器会把连续汉字当成一个整词，中文几乎检索不到。这里把中文
    展开成【单字 + bigram】再入库（查询同样处理），让 BM25 在中文上真正生效。
"""
import re
import sqlite3
import time

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    source     TEXT NOT NULL,
    text       TEXT NOT NULL,
    embed_mode TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_kb_doc ON kb_chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(chunk_id UNINDEXED, text);
"""

# FTS 查询参与匹配的 term 上限，避免 SQL 过长
_FTS_MAX_TERMS = 60

# Milvus Lite 客户端（懒加载：只有真正用到向量时才初始化）
_MV = None


def _conn() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(str(config.RAG_DB))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _mv():
    global _MV
    if _MV is None:
        from pymilvus import MilvusClient
        config.ensure_dirs()
        _MV = MilvusClient(str(config.MILVUS_DB))
    return _MV


def _coll_name(embed_mode: str, dim: int) -> str:
    """collection 名带模式与维度：换嵌入模型时新旧向量天然隔离。"""
    return "kb_%s_%d" % (embed_mode, int(dim))


def _ensure_coll(client, name: str, dim: int) -> str:
    if not client.has_collection(name):
        client.create_collection(
            collection_name=name,
            dimension=int(dim),
            primary_field_name="id",
            id_type="string",
            auto_id=False,
            vector_field_name="vector",
            metric_type="COSINE",
        )
    _load(client, name)
    return name


def _load(client, name: str) -> None:
    """Milvus 搜索前必须 load；进程重启后 collection 会回到 released，需重新 load。"""
    try:
        client.load_collection(name)
    except Exception:  # noqa: BLE001  已 load 或并发 load 都会抛，忽略
        pass


def _kb_collections(client) -> list:
    try:
        return [n for n in client.list_collections() if n.startswith("kb_")]
    except Exception:  # noqa: BLE001
        return []


def _log(event: str, payload: dict) -> None:
    """向量库异常不抛出：检索会退化为只用 BM25，但要留痕。"""
    try:
        import logging_utils as telemetry
        telemetry.event(event, payload)
    except Exception:  # noqa: BLE001
        pass


def fts_encode(text: str) -> str:
    """把文本编码成 FTS5 可检索的形式：中文展开为单字 + bigram。"""
    toks = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text or "")
    han = re.findall(r"[\u4e00-\u9fff]", text or "")
    toks.extend(a + b for a, b in zip(han, han[1:]))
    return " ".join(toks)


def count() -> int:
    """片段总数以 SQLite 为准（向量库只做检索，不作为计数来源）。"""
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) c FROM kb_chunks").fetchone()["c"]


def clear() -> None:
    """清空检索库（全量重建索引用）：SQLite 与向量库都清。"""
    with _conn() as conn:
        conn.execute("DELETE FROM kb_chunks")
        conn.execute("DELETE FROM kb_fts")
    try:
        client = _mv()
        for name in _kb_collections(client):
            client.drop_collection(name)
    except Exception as e:  # noqa: BLE001
        _log("milvus_error", {"op": "clear", "error": str(e)[:160]})


def delete_doc(doc_id: str) -> None:
    """删除同一文档的全部 chunk：文本、全文索引、向量三处一起删。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, embed_mode, dim FROM kb_chunks WHERE doc_id=?",
            (doc_id,)).fetchall()
        for r in rows:
            conn.execute("DELETE FROM kb_fts WHERE chunk_id=?", (r["chunk_id"],))
        conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))

    if not rows:
        return
    # 向量按 (模式, 维度) 分组，删对应 collection
    groups = {}
    for r in rows:
        groups.setdefault((r["embed_mode"], int(r["dim"])), []).append(r["chunk_id"])
    try:
        client = _mv()
        for (mode, dim), ids in groups.items():
            name = _coll_name(mode, dim)
            if client.has_collection(name):
                client.delete(name, ids=ids)
    except Exception as e:  # noqa: BLE001
        _log("milvus_error", {"op": "delete", "error": str(e)[:160]})


def upsert(doc_id: str, source: str, texts, vectors) -> int:
    """写入一个文档的全部 chunk（先清旧的，保证重复索引幂等）。

    顺序：先 SQLite（文本 + 全文索引，权威），再 Milvus（向量）。
    向量写失败不回滚文本——检索会退化成 BM25，但不会丢数据。
    """
    texts = [t for t in texts if t]
    if not texts:
        return 0
    delete_doc(doc_id)

    mode = config.EMBED_MODE
    dim = len(vectors[0]) if vectors else 0
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    rows, mv_rows = [], []
    for i, (t, v) in enumerate(zip(texts, vectors)):
        cid = "%s#%d" % (doc_id, i)
        rows.append((cid, doc_id, source, t, mode, dim, now))
        if v:
            mv_rows.append({"id": cid, "vector": list(v),
                            "source": source, "doc_id": doc_id})

    with _conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO kb_chunks "
            "(chunk_id, doc_id, source, text, embed_mode, dim, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.executemany(
            "INSERT INTO kb_fts (chunk_id, text) VALUES (?, ?)",
            [(r[0], fts_encode(r[3])) for r in rows])

    if mv_rows and dim:
        try:
            client = _mv()
            name = _ensure_coll(client, _coll_name(mode, dim), dim)
            client.upsert(name, mv_rows)
        except Exception as e:  # noqa: BLE001
            _log("milvus_error", {"op": "upsert", "error": str(e)[:160]})
    return len(rows)


def vector_search(query_vec, embed_mode: str, dim: int, top_k: int = 50):
    """向量检索（Milvus ANN）：返回 [(chunk_id, 余弦相似度)]，按相似度降序。"""
    try:
        client = _mv()
        name = _coll_name(embed_mode, dim)
        if not client.has_collection(name):
            return []
        _load(client, name)   # 进程重启后 collection 可能处于 released
        res = client.search(name, data=[list(query_vec)], limit=int(top_k),
                            output_fields=["id"])
    except Exception as e:  # noqa: BLE001
        _log("milvus_error", {"op": "search", "error": str(e)[:160]})
        return []

    hits = res[0] if res else []
    out = []
    for h in hits:
        cid = h.get("id")
        score = h.get("distance")
        if cid and score is not None and score > 0:   # 只保留正相关
            out.append((cid, float(score)))
    return out


def fts_search(query: str, top_k: int = 50):
    """BM25 全文检索：返回 [(chunk_id, score)]，score 越大越相关。"""
    terms = fts_encode(query).split()[:_FTS_MAX_TERMS]
    if not terms:
        return []
    match = " OR ".join('"%s"' % t.replace('"', "") for t in terms)
    with _conn() as conn:
        try:
            rows = conn.execute(
                "SELECT chunk_id, bm25(kb_fts) AS s FROM kb_fts WHERE kb_fts MATCH ? "
                "ORDER BY s LIMIT ?", (match, top_k)).fetchall()
        except sqlite3.OperationalError:
            return []
    return [(r["chunk_id"], -float(r["s"])) for r in rows]


def get_texts(chunk_ids):
    """按 chunk_id 批量取原文（向量结果与 BM25 结果都要回表取文本）。"""
    if not chunk_ids:
        return {}
    q = ",".join("?" * len(chunk_ids))
    with _conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, source, text FROM kb_chunks "
            "WHERE chunk_id IN (%s)" % q, list(chunk_ids)).fetchall()
    return {r["chunk_id"]: dict(r) for r in rows}
