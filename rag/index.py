"""索引构建：把"记忆与知识"写进检索库。

三种来源（都带 source 标记，检索结果可溯源）：
    memory_fact  用户明确说过的偏好/预算（画像事实）
    memory_turn  过去的问答（问 + 答）
    kb_file      本地知识库文件（data/kb/*.md，可选）

幂等性：doc_id 由内容哈希生成，同一条重复写入只会覆盖，不会堆积重复结果。
静默失败：索引属于"增强"，绝不能因为嵌入接口出错就让用户这次问答失败。
"""
import hashlib
import re

import config
import logging_utils as telemetry
from memory import store as mem
from rag import chunk as chunker, embed, store as kb


# 引用编号：【记忆:xxx】/ [记忆:xxx]
_CITE_RE = re.compile(r"[【\[]记忆[:：][^】\]]*[】\]]")


def _doc_id(prefix: str, text: str) -> str:
    return "%s:%s" % (prefix, hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12])


def strip_citations(text: str) -> str:
    """入库前剥掉引用编号，避免"引用传染"。

    历史回答里带着 [记忆:xxx]，一旦被索引，下次检索到这段文本时模型会连着
    旧引用一起复述——于是引用编号指向的并不是本轮证据，属于误导。
    清洗后再入库，引用就只可能来自本轮检索结果。
    """
    return _CITE_RE.sub("", text or "").strip()


def add_document(doc_id: str, source: str, text: str,
                 size: int = None, overlap: int = None) -> int:
    """切分 → 嵌入 → 入库，返回写入的 chunk 数。

    size/overlap 可覆盖默认切分粒度（消融实验用它对比不同切分大小的效果）。
    """
    text = strip_citations(text)
    kw = {}
    if size:
        kw["size"] = size
    if overlap is not None:
        kw["overlap"] = overlap
    chunks = chunker.split_text(text, **kw)
    if not chunks:
        return 0
    vecs = embed.embed_many(chunks)
    if not vecs or len(vecs) != len(chunks):
        return 0
    return kb.upsert(doc_id, source, chunks, vecs)


def index_fact(fact: str, size: int = None) -> int:
    fact = (fact or "").strip()
    if not fact:
        return 0
    return add_document(_doc_id("fact", fact), "memory_fact", fact, size=size)


def index_turn(question: str, answer: str, size: int = None) -> int:
    text = "问：%s\n答：%s" % ((question or "").strip(), (answer or "").strip())
    if len(text) <= 4:
        return 0
    return add_document(_doc_id("turn", text), "memory_turn", text, size=size)


def index_kb_files(size: int = None) -> int:
    """索引 data/kb/*.md（可选本地知识库；目录不存在则跳过）。"""
    try:
        files = sorted(config.KB_DIR.glob("*.md"))
    except Exception:  # noqa: BLE001
        return 0
    total = 0
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if text.strip():
            total += add_document(_doc_id("kb", p.name), "kb_file", text, size=size)
    return total


def reindex_all(limit_turns: int = 100, size: int = None) -> dict:
    """全量重建索引（CLI: main.py --index）。"""
    kb.clear()
    stats = {"fact_docs": 0, "turn_docs": 0, "kb_chunks": 0,
             "chunks": 0, "embed_mode": config.EMBED_MODE, "chunk_size": size}
    for f in mem.get_facts(limit=500):
        n = index_fact(f.get("fact", ""), size=size)
        if n:
            stats["fact_docs"] += 1
            stats["chunks"] += n
    for t in mem.recent_turns(limit=limit_turns):
        n = index_turn(t.get("question", ""), t.get("answer", ""), size=size)
        if n:
            stats["turn_docs"] += 1
            stats["chunks"] += n
    stats["kb_chunks"] = index_kb_files(size=size)
    stats["chunks"] += stats["kb_chunks"]
    telemetry.event("rag_index", stats)
    return stats


def ensure_indexed() -> bool:
    """库为空时惰性构建一次，保证"装好就能用"，不要求用户先跑命令。"""
    if kb.count() > 0:
        return False
    reindex_all()
    return True


# ---------------------------------------------------------------------------
# 运行期钩子：任何异常都只记录、不抛出——索引失败不能影响主流程
# ---------------------------------------------------------------------------

def _safe(fn, *args, **kwargs) -> int:
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        telemetry.event("rag_index_error", {"error": str(e)[:160]})
        return 0


def try_index_fact(fact: str) -> int:
    if not config.RAG_AUTO_INDEX:
        return 0
    return _safe(index_fact, fact)


def try_index_turn(question: str, answer: str) -> int:
    if not config.RAG_AUTO_INDEX:
        return 0
    return _safe(index_turn, question, answer)
