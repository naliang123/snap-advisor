"""离线评测：工具触发正确率 / 无证据断言 / 价格引用。

默认强制 mock 模式（在 import config 之前设置环境变量），保证结果可复现；
评测使用独立的 eval_memory.db，不污染日常使用的记忆库。

用法：
  python evals/run_eval.py           # 离线 mock 评测（推荐）
  python evals/run_eval.py --real    # 使用当前 .env 的真实模式（结果不可复现）
输出：终端表格 + data/eval_report.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(real: bool = False):
    # Windows GBK 控制台兜底
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    if not real:
        os.environ["LLM_MODE"] = "mock"
        os.environ["VLM_MODE"] = "mock"
        os.environ["PRICE_MODE"] = "mock"
        os.environ["SNAP_MEMORY_DB"] = str(ROOT / "data" / "eval_memory.db")
    sys.path.insert(0, str(ROOT))

    import config  # noqa: E402
    from agent import runner  # noqa: E402

    runner.ensure_sample()
    cases = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        image = str(ROOT / case["image"]) if case.get("image") else None
        result = runner.run_once(case["question"], image)

        triggered = [t["tool"] for t in result.get("tool_results", [])]
        answer = result.get("final_answer") or ""
        violations = result.get("violations") or []

        tools_ok = set(triggered) >= set(case.get("expect_tools", []))
        mentions_ok = all(p in answer for p in case.get("expect_mentions", []))

        rows.append({
            "id": case["id"],
            "tools_triggered": triggered,
            "tools_ok": tools_ok,
            "mentions_ok": mentions_ok,
            "violations": violations,
            "answer_head": answer[:40],
        })

    # 汇总
    n = len(rows)
    tools_rate = sum(1 for r in rows if r["tools_ok"]) / n if n else 0
    mention_rate = sum(1 for r in rows if r["mentions_ok"]) / n if n else 0
    total_viol = sum(len(r["violations"]) for r in rows)

    print("=" * 66)
    print("离线评测报告（模式: LLM=%s VLM=%s 比价=%s）" % (
        config.LLM_MODE, config.VLM_MODE, config.PRICE_MODE))
    print("=" * 66)
    print("%-16s %-28s %-8s %-8s %s" % ("用例", "触发工具", "工具", "引用", "违规"))
    for r in rows:
        print("%-16s %-28s %-8s %-8s %d" % (
            r["id"], ",".join(r["tools_triggered"]) or "-",
            "OK" if r["tools_ok"] else "FAIL",
            "OK" if r["mentions_ok"] else "FAIL",
            len(r["violations"])))
    print("-" * 66)
    print("工具触发正确率: %d/%d (%.0f%%)" % (sum(1 for r in rows if r["tools_ok"]), n, tools_rate * 100))
    print("价格引用正确率: %d/%d (%.0f%%)" % (sum(1 for r in rows if r["mentions_ok"]), n, mention_rate * 100))
    print("无证据断言总数: %d（严格模式下应为 0）" % total_viol)

    report = {"summary": {"tools_rate": tools_rate, "mention_rate": mention_rate,
                          "violations": total_viol}, "cases": rows}
    out = ROOT / "data" / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告已写入: %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="使用真实模式（不可复现）")
    main(real=ap.parse_args().real)
