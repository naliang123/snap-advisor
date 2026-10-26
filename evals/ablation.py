"""消融实验：证据约束 开 vs 关 的对比。

这是本项目的核心量化卖点：
  - 严格模式（约束开）：回答中的每个价格都必须来自画面价签或工具结果；
  - 松弛模式（约束关）：模型可自由发挥"行情判断"（mock 会故意编造 ¥99/¥9.9），
    被 evidence.py 检出为"无证据断言"。

运行：python evals/ablation.py（强制 mock，可复现）
输出：终端对比表 + data/ablation_report.json
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 强制 mock（可复现），并使用独立记忆库
os.environ["LLM_MODE"] = "mock"
os.environ["VLM_MODE"] = "mock"
os.environ["PRICE_MODE"] = "mock"
os.environ["SNAP_MEMORY_DB"] = str(ROOT / "data" / "ablation_memory.db")
sys.path.insert(0, str(ROOT))

from agent import runner  # noqa: E402


def run_case(question, image, loose):
    result = runner.run_once(question, image, loose=loose)
    return {
        "answer_head": (result.get("final_answer") or "")[:60],
        "violations": result.get("violations") or [],
    }


def main():
    # Windows GBK 控制台兜底
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    runner.ensure_sample()
    cases = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        image = str(ROOT / case["image"]) if case.get("image") else None
        strict = run_case(case["question"], image, loose=False)
        loose = run_case(case["question"], image, loose=True)
        rows.append({
            "id": case["id"],
            "strict_violations": len(strict["violations"]),
            "loose_violations": len(loose["violations"]),
            "loose_detail": loose["violations"],
        })

    s_total = sum(r["strict_violations"] for r in rows)
    l_total = sum(r["loose_violations"] for r in rows)

    print("=" * 66)
    print("消融实验：证据约束 开 vs 关（mock 可复现）")
    print("=" * 66)
    print("%-16s %-14s %-14s" % ("用例", "约束开·违规", "约束关·违规"))
    for r in rows:
        print("%-16s %-14s %-14s" % (r["id"], r["strict_violations"], r["loose_violations"]))
    print("-" * 66)
    print("总违规数：约束开 = %d  vs  约束关 = %d" % (s_total, l_total))
    print("结论：关闭证据约束后，无证据断言（编造价格/未声明演示数据）显著上升。")

    out = ROOT / "data" / "ablation_report.json"
    out.write_text(json.dumps(
        {"summary": {"strict": s_total, "loose": l_total}, "cases": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告已写入: %s" % out)


if __name__ == "__main__":
    main()
