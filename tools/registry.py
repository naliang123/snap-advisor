"""Agent 工具注册表：OpenAI function-calling schema + 本地分发。

设计原则：每个工具的返回值都带 evidence（来源 + 时间 + 是否演示），
供"诚实约束"（回答只能引用有证据的信息）与评测使用。
"""
import json
import time

from memory import store
from rag import index as rag_index
from tools import kb, price

# OpenAI function calling 工具 schema（真实模式直接下发给对话模型）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compare_price",
            "description": "线上比价：给定商品名，返回各平台报价与证据。用于判断线下价格是否划算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "商品名，例如：保温杯"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "把用户明确表达的偏好/预算/忌口写入长期画像（SQLite）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "要记住的偏好，一句话"}
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_profile",
            "description": "查询当前用户画像（已记住的偏好列表）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "在用户自己的记忆（说过的偏好、过去的问答）与本地知识库里检索相关"
                "内容。用于回忆用户以前说过什么、以前问过什么；资料变多时比全量"
                "列出更高效。检索不到就如实说没查到。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "检索问题，例如：我以前说过买洗发水要注意什么"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_tool(name: str, arguments_json: str) -> dict:
    """执行工具调用，返回带 evidence 的 dict（写入 tool message 的内容）。"""
    try:
        args = json.loads(arguments_json or "{}")
    except Exception:
        args = {}

    if name == "compare_price":
        return price.search_price_offers(args.get("query", ""))

    if name == "remember_preference":
        fact = (args.get("fact") or "").strip()
        if not fact:
            return {"status": "empty", "reason": "没有可记住的内容"}
        store.add_fact(fact, source="user_statement")
        # 同步进检索库：以后问起相关话题时能被召回（失败不影响本次记住）
        rag_index.try_index_fact(fact)
        return {"status": "已记住", "fact": fact,
                "evidence": {"source": "user_statement", "fetched_at": _now()}}

    if name == "query_profile":
        facts = [f["fact"] for f in store.get_facts()]
        return {"facts": facts,
                "evidence": {"source": "memory", "fetched_at": _now()}}

    if name == "search_kb":
        return kb.search_kb(args.get("query", ""), int(args.get("top_k", 5) or 5))

    return {"error": "未知工具 %s" % name}
