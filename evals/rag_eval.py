"""检索评测：召回率 / 排序质量 / 端到端是否真的用上了检索。

离线 mock 可复现（默认）。注意：mock 嵌入是【词面】匹配，语义泛化能力只有
配置 EMBED_API_KEY 后才会真正体现——表格里会把 embedding 模式一并打出，
不拿伪向量的成绩冒充语义检索。

用法：
  python evals/rag_eval.py           # 离线 mock（推荐，可复现）
  python evals/rag_eval.py --real    # 真实 embedding（需 EMBED_API_KEY）
输出：终端表格 + data/rag_eval_report.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 5


def _reset():
    """评测用独立的记忆库与检索库，且每次清空，保证结果可复现。"""
    for name in ("eval_memory.db", "eval_rag.db"):
        p = ROOT / "data" / name
        if p.exists():
            p.unlink()


def _seed(mem, data):
    """把"过去的记忆"写进记忆库（检索库随后从记忆库重建）。"""
    for f in data["seed"]["facts"]:
        mem.add_fact(f)
    for t in data["seed"]["turns"]:
        mem.log_turn(t["question"], t["answer"], "", "[]")


def _seed_long_docs(ri, data):
    """长文档单独入库：它们不在记忆库里，是"本地知识库"的角色。

    必须放在 reindex_all() 之后——全量重建会先清空检索库。
    """
    for d in data.get("long_docs", []):
        ri.add_document("kb:" + d["id"], "kb_file", d["content"])
    return len(data.get("long_docs", []))


def _rank_of(hits, keywords):
    """返回第一条同时包含全部关键词的排名（1 起，0 表示没召回）。"""
    for i, h in enumerate(hits, 1):
        if all(k in h["text"] for k in keywords):
            return i
    return 0


def main(real: bool = False):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    _reset()
    if not real:
        os.environ["LLM_MODE"] = "mock"
        os.environ["VLM_MODE"] = "mock"
        os.environ["PRICE_MODE"] = "mock"
    os.environ["SNAP_MEMORY_DB"] = str(ROOT / "data" / "eval_memory.db")
    os.environ["SNAP_RAG_DB"] = str(ROOT / "data" / "eval_rag.db")
    sys.path.insert(0, str(ROOT))

    import config  # noqa: E402
    from agent import runner  # noqa: E402
    from memory import store as mem  # noqa: E402
    from rag import index as ri, retrieve, store as kbstore  # noqa: E402

    data = json.loads((ROOT / "evals" / "rag_dataset.json").read_text(encoding="utf-8"))
    _seed(mem, data)
    stats = ri.reindex_all()
    stats["kb_docs"] = _seed_long_docs(ri, data)
    stats["chunks"] = kbstore.count()

    rows = []
    for case in data["cases"]:
        hits = retrieve.search(case["question"], top_k=TOP_K)
        rank = _rank_of(hits, case["expect"])
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "rank": rank,
            "hit@1": rank == 1,
            "hit@%d" % TOP_K: 0 < rank <= TOP_K,
            "rr": (1.0 / rank) if rank else 0.0,
            "top1": (hits[0]["text"][:34] if hits else ""),
        })

    n = len(rows)
    recall_at_k = sum(1 for r in rows if r["hit@%d" % TOP_K]) / n
    recall_at_1 = sum(1 for r in rows if r["hit@1"]) / n
    mrr = sum(r["rr"] for r in rows) / n

    # ---- 端到端：Agent 是否真的会主动用检索（而不只是检索器本身能跑）----
    runner.ensure_sample()
    e2e = []
    for q in data.get("e2e", []):
        res = runner.run_once(q, str(config.SAMPLE_IMAGE))
        tools = [t["tool"] for t in res.get("tool_results", [])]
        ans = res.get("final_answer") or ""
        e2e.append({"question": q, "tools": tools,
                    "used_kb": "search_kb" in tools,
                    "cited": ("[记忆:" in ans or "【记忆:" in ans),
                    "violations": res.get("violations") or []})
    e2e_rate = (sum(1 for x in e2e if x["used_kb"]) / len(e2e)) if e2e else 0.0

    print("=" * 74)
    print("检索评测（embedding=%s  检索=%s  重排=%s）" % (
        config.EMBED_MODE, config.RAG_MODE, "开" if config.RAG_RERANK else "关"))
    print("索引：%d 个文档 / %d 个片段" % (
        stats["fact_docs"] + stats["turn_docs"], stats["chunks"]))
    print("=" * 74)
    print("%-6s %-30s %-6s %s" % ("用例", "问题", "排名", "Top1 命中内容"))
    for r in rows:
        print("%-6s %-30s %-6s %s" % (
            r["id"], r["question"][:28], r["rank"] or "-", r["top1"]))
    print("-" * 74)
    print("Recall@%d: %.0f%%    Recall@1: %.0f%%    MRR: %.3f" % (
        TOP_K, recall_at_k * 100, recall_at_1 * 100, mrr))
    cite_viol = sum(len(x["violations"]) for x in e2e)
    print("端到端：%d/%d 个回忆类问题主动触发了 search_kb（%.0f%%）" % (
        sum(1 for x in e2e if x["used_kb"]), len(e2e), e2e_rate * 100))
    print("        回答里的引用编号全部经 grounding 校验，违规 %d 处（应为 0）" % cite_viol)
    for x in e2e:
        print("   [%s] %s  工具=%s  引用=%s  违规=%d" % (
            "触发" if x["used_kb"] else "未触发", x["question"][:24],
            ",".join(x["tools"]) or "-", "有" if x["cited"] else "无",
            len(x["violations"])))

    report = {
        "mode": {"embed_mode": config.EMBED_MODE, "rag_mode": config.RAG_MODE,
                 "rerank": config.RAG_RERANK},
        "index": stats,
        "summary": {"recall@%d" % TOP_K: recall_at_k, "recall@1": recall_at_1,
                    "mrr": mrr, "e2e_trigger_rate": e2e_rate},
        "cases": rows,
        "e2e": e2e,
    }
    out = ROOT / "data" / "rag_eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告已写入: %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="真实 embedding（不可复现）")
    main(real=ap.parse_args().real)
