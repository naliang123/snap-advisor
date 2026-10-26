"""配置中心：集中读取 .env / 环境变量。

设计原则：所有默认值保证【零 API Key 也能完整运行】——LLM / VLM / 比价
在未配置 Key 时自动落入离线 mock 模式，Agent 全链路（含 function calling）
可复现地跑通，便于开发、测试与评测。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# load_dotenv 默认【不覆盖】已存在的环境变量，因此测试/评测脚本可以先设置
# LLM_MODE=mock 等变量再 import config，从而强制离线模式。
load_dotenv(ROOT / ".env")


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _auto_mode(mode_env: str, key_env: str) -> str:
    """显式指定 mock/real 优先；否则按是否配置了对应 Key 自动判定。"""
    explicit = os.getenv(mode_env, "").strip().lower()
    if explicit in ("mock", "real"):
        return explicit
    return "real" if os.getenv(key_env, "").strip() else "mock"


# ---------- 对话 LLM（任意 OpenAI 兼容接口，默认 DeepSeek） ----------
LLM_MODE = _auto_mode("LLM_MODE", "DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# ---------- 视觉 VLM（任意 OpenAI 兼容多模态接口，默认通义 qwen-vl） ----------
VLM_MODE = _auto_mode("VLM_MODE", "VLM_API_KEY")
VLM_API_KEY = os.getenv("VLM_API_KEY", "").strip()
VLM_BASE_URL = os.getenv(
    "VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-plus").strip()

# ---------- 比价工具 ----------
PRICE_MODE = os.getenv("PRICE_MODE", "mock").strip().lower()  # mock | web
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "8"))

# ---------- 检索（RAG：把记忆与知识变成可召回的证据）----------
# 未配置 EMBED_API_KEY 时自动落入 mock（词面哈希伪向量），零 Key 仍可跑通；
# 想要真正的语义检索，需另配 embedding 服务（与对话 LLM 可以是不同供应商）。
EMBED_MODE = _auto_mode("EMBED_MODE", "EMBED_API_KEY")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "").strip()
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
RAG_MODE = os.getenv("RAG_MODE", "hybrid").strip().lower()   # vector | bm25 | hybrid
RAG_RERANK = _flag("RAG_RERANK", "0")            # 是否对候选做重排（真实模式走 LLM）
RAG_AUTO_INDEX = _flag("RAG_AUTO_INDEX", "1")    # 问答结束后自动增量索引

# ---------- 运行行为 ----------
CONFIRM_MODE = _flag("SNAP_CONFIRM", "0")        # 购买建议前人工确认（HITL）
EVIDENCE_STRICT = not _flag("SNAP_LOOSE", "0")   # 证据约束总开关（消融实验用）
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "3"))  # 单轮问答最大工具调用轮数

# ---------- 路径（均可通过环境变量重定向，便于测试隔离） ----------
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PERCEPTIONS_DIR = DATA_DIR / "perceptions"
MEMORY_DB = Path(os.getenv("SNAP_MEMORY_DB", str(DATA_DIR / "memory.db")))
RAG_DB = Path(os.getenv("SNAP_RAG_DB", str(DATA_DIR / "rag.db")))
TRACE_FILE = Path(os.getenv("SNAP_TRACE_FILE", str(DATA_DIR / "trace.jsonl")))
SAMPLE_IMAGE = DATA_DIR / "sample_shelf.jpg"
KB_DIR = DATA_DIR / "kb"          # 可选本地知识库（放 *.md 即被索引）


def ensure_dirs() -> None:
    """创建运行期目录（幂等）。"""
    for p in (DATA_DIR, UPLOAD_DIR, PERCEPTIONS_DIR, KB_DIR,
              MEMORY_DB.parent, RAG_DB.parent):
        p.mkdir(parents=True, exist_ok=True)


def mode_summary() -> dict:
    """当前各层模式的汇总（CLI 横幅 / 网页角标共用）。"""
    return {
        "llm_mode": LLM_MODE,
        "vlm_mode": VLM_MODE,
        "price_mode": PRICE_MODE,
        "evidence_strict": EVIDENCE_STRICT,
        "confirm_mode": CONFIRM_MODE,
        "llm_model": DEEPSEEK_MODEL if LLM_MODE == "real" else "offline-mock",
        "vlm_model": VLM_MODEL if VLM_MODE == "real" else "offline-mock",
        "embed_mode": EMBED_MODE,
        "rag_mode": RAG_MODE,
    }
