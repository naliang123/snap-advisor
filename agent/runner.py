"""统一运行入口：CLI / Web / 评测共用，避免三处逻辑漂移。

run_once(question, image_path, loose, messages)：
  - messages 传入时会作为历史继续（多轮对话），并追加本轮 user 消息；
  - 返回完整的最终 state（final_answer / perception / tool_results / violations）。
"""
import config
from agent.graph import get_graph
from memory import store
from rag import index as rag_index


def run_once(question: str, image_path=None, loose=None, messages=None) -> dict:
    graph = get_graph()
    facts = store.get_facts()
    history = list(messages) if messages else []
    state = {
        "question": question,
        "image_path": str(image_path) if image_path else None,
        "messages": history + [{"role": "user", "content": question}],
        "profile_facts": facts,
        "tool_results": [],
        "tool_rounds": 0,
        "confirmed": False,
        "regenerate": False,
        "loose": loose if loose is not None else (not config.EVIDENCE_STRICT),
        "trace": [],
        "perception": None,
        "violations": [],
    }
    result = graph.invoke(state, {"recursion_limit": 40})
    # 本轮问答写入检索库：以后问到相关话题时能被召回（失败不影响本次结果）
    if config.RAG_AUTO_INDEX and result.get("final_answer"):
        rag_index.try_index_turn(question, result["final_answer"])
    return result


def ensure_sample() -> None:
    """示例货架图或标注缺失时，自动运行生成脚本（幂等）。"""
    sidecar = config.PERCEPTIONS_DIR / (config.SAMPLE_IMAGE.name + ".json")
    if config.SAMPLE_IMAGE.exists() and sidecar.exists():
        return
    import importlib.util
    path = config.ROOT / "scripts" / "make_sample_image.py"
    spec = importlib.util.spec_from_file_location("make_sample_image", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
