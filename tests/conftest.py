"""pytest 全局夹具：在任何模块 import config 之前强制离线 mock 并隔离数据。"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 必须先设环境变量再被业务模块 import（load_dotenv 不覆盖已有变量）
os.environ["LLM_MODE"] = "mock"
os.environ["VLM_MODE"] = "mock"
os.environ["PRICE_MODE"] = "mock"
# 嵌入也必须锁 mock：否则配了 EMBED_API_KEY 的 .env 会让测试去调真实嵌入服务，
# 既变慢、要花钱，又因为网络抖动导致结果不可复现。
os.environ["EMBED_MODE"] = "mock"

_tmp = tempfile.mkdtemp(prefix="snap_test_")
os.environ["SNAP_MEMORY_DB"] = os.path.join(_tmp, "memory.db")
os.environ["SNAP_RAG_DB"] = os.path.join(_tmp, "rag.db")
os.environ["SNAP_TRACE_FILE"] = os.path.join(_tmp, "trace.jsonl")

sys.path.insert(0, str(ROOT))
