"""检索消融实验：用数据回答"这样做到底值不值"。

三组实验（每组都回答一个真问题）：
  A 检索策略：纯向量 / 纯关键词(BM25) / 混合 / 混合+重排，谁更好？
  B 切分大小：200 / 500 / 1000 字，切多大召回最好？
  C 注入方式：全部记忆塞进提示词 vs 只塞检索到的 5 条——为什么必须检索？

用法：
  python evals/rag_ablation.py
输出：终端三张表 + data/rag_ablation_report.json
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 5
# 中文 token 粗略估算系数（用于量级对比，不是精确计费口径）
CHARS_PER_TOKEN = 1.6
CONTEXT_WINDOW = 8000


def _reset():
    for name in ("ablation_memory.db", "ablation_rag.db"):
        p = ROOT / "data" / name
        if p.exists():
            p.unlink()


def _measure(cases, retrieve, mode=None, rerank=False, top_k=TOP_K):
    hit_k = hit_1 = 0
    mrr = 0.0
    for case in cases:
        hits = retrieve.search(case["question"], top_k=top_k, mode=mode, rerank=rerank)
        rank = 0
        for i, h in enumerate(hits, 1):
            if all(k in h["text"] for k in case["expect"]):
                rank = i
                break
        if rank:
            mrr += 1.0 / rank
            hit_k += 1
            hit_1 += (rank == 1)
    n = len(cases) or 1
    return {"recall@%d" % top_k: hit_k / n, "recall@1": hit_1 / n, "mrr": mrr / n}


def _pct(x):
    return "%.0f%%" % (x * 100)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    _reset()
    os.environ["LLM_MODE"] = "mock"
    os.environ["VLM_MODE"] = "mock"
    os.environ["PRICE_MODE"] = "mock"
    os.environ["SNAP_MEMORY_DB"] = str(ROOT / "data" / "ablation_memory.db")
    os.environ["SNAP_RAG_DB"] = str(ROOT / "data" / "ablation_rag.db")
    sys.path.insert(0, str(ROOT))

    import config  # noqa: E402
    from memory import store as mem  # noqa: E402
    from rag import index as ri, retrieve, store as kbstore  # noqa: E402

    data = json.loads((ROOT / "evals" / "rag_dataset.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    for f in data["seed"]["facts"]:
        mem.add_fact(f)
    for t in data["seed"]["turns"]:
        mem.log_turn(t["question"], t["answer"], "", "[]")

    def build(size=None):
        """重建索引：记忆来自记忆库，长文档单独入库（全量重建会先清空）。"""
        ri.reindex_all(size=size)
        for d in data.get("long_docs", []):
            ri.add_document("kb:" + d["id"], "kb_file", d["content"], size=size)
        return kbstore.count()

    report = {"embed_mode": config.EMBED_MODE, "experiments": {}}

    # ---------------- A 检索策略 ----------------
    build()
    plans = [("纯向量", "vector", False), ("纯关键词 BM25", "bm25", False),
             ("混合 RRF", "hybrid", False), ("混合+重排", "hybrid", True)]
    print("=" * 74)
    print("A 检索策略对比（embedding=%s，离线为词面伪向量）" % config.EMBED_MODE)
    print("=" * 74)
    print("%-16s %-10s %-10s %s" % ("策略", "Recall@5", "Recall@1", "MRR"))
    rows_a = {}
    for name, mode, rr in plans:
        m = _measure(cases, retrieve, mode=mode, rerank=rr)
        rows_a[name] = m
        print("%-16s %-10s %-10s %.3f" % (
            name, _pct(m["recall@%d" % TOP_K]), _pct(m["recall@1"]), m["mrr"]))
    print("（离线 mock 下三者差距小是预期内的：伪向量只比词面重合，"
          "真实语义优势需配置 EMBED_API_KEY 后再看）")
    report["experiments"]["strategy"] = rows_a

    # ---------------- B 切分大小 ----------------
    print("\n" + "=" * 74)
    print("B 切分大小对比（检索=%s）" % config.RAG_MODE)
    print("=" * 74)
    print("%-12s %-10s %-10s %-10s %s" % ("切分字数", "片段数", "Recall@5", "Recall@1", "MRR"))
    rows_b = {}
    for size in (200, 500, 1000):
        chunks = build(size=size)
        m = _measure(cases, retrieve)
        rows_b[str(size)] = dict(m, chunks=chunks)
        print("%-12s %-10s %-10s %-10s %.3f" % (
            size, chunks, _pct(m["recall@%d" % TOP_K]),
            _pct(m["recall@1"]), m["mrr"]))
    print("（切太小丢上下文、切太大稀释相关度——上表就是这个取舍的实测结果）")
    report["experiments"]["chunk_size"] = rows_b

    # ---------------- C 注入方式 ----------------
    ri.reindex_all()
    texts = list(data["seed"]["facts"]) + [
        "问：%s\n答：%s" % (t["question"], t["answer"]) for t in data["seed"]["turns"]]
    avg_len = sum(len(t) for t in texts) / len(texts)
    print("\n" + "=" * 74)
    print("C 注入方式对比：全量注入 vs 只注入检索到的 Top-5")
    print("    （token 为粗略估算：%.1f 字符≈1 token，用于量级对比）" % CHARS_PER_TOKEN)
    print("=" * 74)
    print("%-10s %-14s %-14s %s" % ("记忆条数", "全量注入 token", "检索注入 token", "全量是否放得下 8k 窗口"))
    rows_c = {}
    for n in (20, 100, 300, 1000):
        full = avg_len * n / CHARS_PER_TOKEN
        retr = avg_len * TOP_K / CHARS_PER_TOKEN
        fits = "是" if full <= CONTEXT_WINDOW else "否（必然被截断）"
        rows_c[str(n)] = {"full_tokens": round(full), "retrieved_tokens": round(retr),
                          "fits_8k": full <= CONTEXT_WINDOW}
        print("%-10s %-14s %-14s %s" % (n, round(full), round(retr), fits))
    print("（记忆越多，全量注入越不可行——这正是把记忆升级成检索的原因）")
    report["experiments"]["injection"] = rows_c

    out = ROOT / "data" / "rag_ablation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n报告已写入: %s" % out)


if __name__ == "__main__":
    main()
