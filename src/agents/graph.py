from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.nodes.compliance_node import (
    compliance_checker_node,
    policy_retrieval_node,
)
from src.agents.nodes.example_node import analyze_node, respond_node
from src.agents.state import AgentState


async def should_continue(state: AgentState) -> str:
    """Route based on whether an error occurred during analysis."""
    if state.get("error"):
        return END
    return "respond"


async def check_compliance_route(state: AgentState) -> str:
    """Route dựa trên lỗi hoặc kết quả compliance."""
    if state.get("error"):
        return END
    return "compliance_check"


def build_graph() -> CompiledStateGraph:
    """Xây dựng đồ thị LangGraph mặc định."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("respond", respond_node)

    # Add edges
    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", should_continue)
    graph.add_edge("respond", END)

    return graph.compile()


def build_compliance_graph() -> CompiledStateGraph:
    """Xây dựng đồ thị LangGraph với RAG Policy Retrieval & Post-generation Compliance Check."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("compliance_check", compliance_checker_node)

    # Add edges
    graph.set_entry_point("policy_retrieval")
    graph.add_edge("policy_retrieval", "analyze")
    graph.add_conditional_edges("analyze", check_compliance_route)
    graph.add_edge("compliance_check", END)

    return graph.compile()


agent = build_graph()
compliance_agent = build_compliance_graph()
