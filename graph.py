"""LangGraph orchestration: sequential Data Retriever -> Report Generator."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import build_retriever_agent, run_report_generator


class PipelineState(TypedDict):
    """Shared state passed between the two agent nodes."""

    query: str
    snippets: str
    answer: str


def retrieve_node(state: PipelineState) -> dict:
    """Node 1: the Data Retriever agent calls its search tool."""
    agent = build_retriever_agent()
    result = agent.invoke({"messages": [HumanMessage(content=state["query"])]})
    snippets = result["messages"][-1].text
    return {"snippets": snippets}


def generate_node(state: PipelineState) -> dict:
    """Node 2: the Report Generator synthesizes the final answer."""
    answer = run_report_generator(state["query"], state["snippets"])
    return {"answer": answer}


def build_graph():
    """Wire the sequential workflow: START -> retrieve -> generate -> END."""
    graph = StateGraph(PipelineState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
