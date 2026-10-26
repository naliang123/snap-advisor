"""LangGraph 状态图装配。

    START → perceive → agent ⇄ tools
                        ↓
                     confirm → (regenerate) → agent
                        ↓
                     finalize → END
"""
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.state import AgentState

_graph = None


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("perceive", nodes.perceive_node)
    g.add_node("agent", nodes.agent_node)
    g.add_node("tools", nodes.tools_node)
    g.add_node("confirm", nodes.confirm_node)
    g.add_node("finalize", nodes.finalize_node)

    g.add_edge(START, "perceive")
    g.add_edge("perceive", "agent")
    g.add_conditional_edges(
        "agent", nodes.route_after_agent,
        {"tools": "tools", "confirm": "confirm"},
    )
    g.add_edge("tools", "agent")
    g.add_conditional_edges(
        "confirm", nodes.route_after_confirm,
        {"agent": "agent", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


def get_graph():
    """单例编译（编译开销只付一次）。"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
