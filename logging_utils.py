"""JSONL 轨迹日志：Agent 每个节点的事件各写一行，供调试与复盘。

失败不影响主流程（静默吞掉 IO 异常）。
"""
import json
import time

import config


def event(node: str, detail: dict) -> None:
    config.ensure_dirs()
    line = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "node": node}
    line.update(detail)
    try:
        with open(config.TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass
