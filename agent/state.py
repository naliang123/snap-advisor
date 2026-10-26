"""LangGraph 全局状态定义。

字段说明：
- messages:      OpenAI 消息序列（含 system/user/assistant/tool），跨节点累积
- perception:    VLM 结构化感知结果（证据来源之一）
- profile_facts: 用户画像（从 SQLite 读取，注入系统提示）
- tool_results:  工具原始输出列表（供证据校验与前端展示）
- tool_rounds:   已执行的工具轮数（防死循环，超过上限后不再下发工具）
- loose:         消融开关——True 表示关闭证据约束（演示无证据断言率上升）
- confirmed:     HITL 确认状态
- regenerate:    用户拒绝草稿建议后，要求重新生成（仅陈述事实）
- final_answer:  最终回答
- violations:    证据校验违规清单（无证据价格 / 未声明演示数据）
- trace:         节点轨迹事件
"""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    image_path: str
    messages: list
    perception: dict
    profile_facts: list
    tool_results: list
    tool_rounds: int
    loose: bool
    confirmed: bool
    regenerate: bool
    final_answer: str
    violations: list
    links: list
    trace: list
