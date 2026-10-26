"""检索库存储：SQLite（chunk 文本 + 向量 BLOB + FTS5 全文索引）。

为什么不引入向量数据库：
    规模在几千条以内时，SQLite + 余弦暴力检索是毫秒级，且零额外依赖、零额外
    服务——符合本项目"每一行都能解释"的取舍。真到十万条、需要 ANN 近似索引时
    再换也不迟，这是可讲的工程判断（而不是"别人都用所以我用"）。

中文全文检索的一个坑：
    SQLite FTS5 的 unicode61 分词器会把连续汉字当成一个整词，导致中文几乎
    检索不到。这里的做法是把中文展开成【单字 + bigram】再入库（查询同样处理），
    用最朴素的方式让 BM25 在中文上真正生效。
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


def _conn() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(str(config.RAG_DB))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def fts_encode(text: str) -> str:
    """把文本编码成 FTS5 可检索的形式：中文展开为单字 + bigram。"""
    toks = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text or "")
    han = re.findall(r"[\u4e00-\u9fff]", text or "")
    toks.extend(a + b for a, b in zip(han, han[1:]))
    return " ".join(toks)


def count() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) c FROM kb_chunks").fetchone()["c"]


def clear() -> None:
    """清空检索库（全量重建索引用）。"""
    with _conn() as conn:
        conn.execute("DELETE FROM kb_chunks")
        conn.execute("DELETE FROM kb_fts")


def delete_doc(doc_id: str) -> None:
    """删除同一文档的全部 chunk（含全文索引），用于幂等覆盖写入。"""
    with _conn() as conn:
        ids = [r["chunk_id"] for r in conn.execute(
            "SELECT chunk_id FROM kb_chunks WHERE doc_id=?", (doc_id,))]
        for cid in ids:
            conn.execute("DELETE FROM kb_fts WHERE chunk_id=?", (cid,))
        conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))


def upsert(doc_id: str, source: str, texts, vectors) -> int:
    """写入一个文档的全部 chunk（先清旧的，保证重复索引幂等）。"""
    texts = [t for t in texts if t]
    if not texts:
        return 0
    delete_doc(doc_id)
    mode = config.EMBED_MODE
    dim = len(vectors[0]) if vectors else 0
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for i, (t, v) in enumerate(zip(texts, vectors)):
        cid = "%s#%d" % (doc_id, i)
        rows.append((cid, doc_id, source, t, mode, dim,
                     _pack(v) if v else None, now))
    with _conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO kb_chunks "
            "(chunk_id, doc_id, source, text, embed_mode, dim, vec, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.executemany(
            "INSERT INTO kb_fts (chunk_id, text) VALUES (?, ?)",
            [(r[0], fts_encode(r[3])) for r in rows])
    return len(rows)


def _pack(vec) -> bytes:
    from rag import embed
    return embed.pack(vec)


def all_vectors(embed_mode: str, dim: int):
    """取出可参与向量检索的全部 chunk（只取同模式同维度的，避免混算）。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, source, text, vec FROM kb_chunks "
            "WHERE embed_mode=? AND dim=? AND vec IS NOT NULL",
            (embed_mode, dim)).fetchall()
    from rag import embed
    return [(r["chunk_id"], r["doc_id"], r["source"], r["text"],
             embed.unpack(r["vec"], dim)) for r in rows]


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
    """按 chunk_id 批量取原文（混合检索的 BM25 侧需要）。"""
    if not chunk_ids:
        return {}
    q = ",".join("?" * len(chunk_ids))
    with _conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, source, text FROM kb_chunks "
            "WHERE chunk_id IN (%s)" % q, list(chunk_ids)).fetchall()
    return {r["chunk_id"]: dict(r) for r in rows}
