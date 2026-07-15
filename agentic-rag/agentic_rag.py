"""
agentic_rag.py

Agentic RAG: retrieval as a decision, not a fixed step.

Instead of unconditionally retrieving on every query, this graph:
  1. Routes each query to decide whether retrieval is needed at all.
  2. Scores retrieval confidence instead of trusting the first pass blindly.
  3. Reformulates and re-queries when confidence is low, up to a retry cap.
  4. Falls back to an honest "I don't know" instead of guessing forever.

This module defines the graph structure and node logic. LLM calls and the
vector store are injected as callables so the graph can be tested with
mocks (see agentic_rag_demo.ipynb) or wired to a real model/vector store
in production.

Companion piece: github.com/abpandey95/multilingual-rag-india
"""

from __future__ import annotations

from typing import Callable, List, Optional, TypedDict

from langgraph.graph import END, StateGraph


class RAGState(TypedDict):
    query: str
    original_query: str
    needs_retrieval: bool
    retrieved_docs: List[str]
    confidence_score: float
    retry_count: int
    max_retries: int
    answer: Optional[str]
    path_taken: List[str]


# ---------------------------------------------------------------------------
# Node factories
#
# Each node is built via a factory that takes the LLM / retriever as a
# dependency. This keeps the graph logic testable without a live model or
# vector store, and swappable across providers.
# ---------------------------------------------------------------------------


def make_router_node(llm_decide: Callable[[str], str]):
    """llm_decide(query) -> 'RETRIEVE' or 'DIRECT'"""

    def router_node(state: RAGState) -> RAGState:
        decision = llm_decide(state["query"]).strip().upper()
        state["needs_retrieval"] = decision == "RETRIEVE"
        state["path_taken"].append("router")
        return state

    return router_node


def make_retrieve_node(vector_search: Callable[[str, int], List[str]], k: int = 5):
    """vector_search(query, k) -> list of doc strings"""

    def retrieve_node(state: RAGState) -> RAGState:
        state["retrieved_docs"] = vector_search(state["query"], k)
        state["path_taken"].append("retrieve")
        return state

    return retrieve_node


def make_confidence_check_node(llm_score: Callable[[str, str], float]):
    """llm_score(query, context) -> float in [0.0, 1.0]"""

    def confidence_check_node(state: RAGState) -> RAGState:
        context = "\n\n".join(state["retrieved_docs"])
        state["confidence_score"] = llm_score(state["query"], context)
        state["path_taken"].append("confidence_check")
        return state

    return confidence_check_node


def make_reformulate_node(llm_reformulate: Callable[[str, str], str]):
    """llm_reformulate(current_query, original_query) -> rewritten query"""

    def reformulate_node(state: RAGState) -> RAGState:
        state["query"] = llm_reformulate(state["query"], state["original_query"])
        state["retry_count"] += 1
        state["path_taken"].append("reformulate")
        return state

    return reformulate_node


def make_direct_answer_node(llm_answer_direct: Callable[[str], str]):
    def direct_answer_node(state: RAGState) -> RAGState:
        state["answer"] = llm_answer_direct(state["query"])
        state["path_taken"].append("direct_answer")
        return state

    return direct_answer_node


def make_generate_answer_node(llm_answer_with_context: Callable[[str, str], str]):
    def generate_answer_node(state: RAGState) -> RAGState:
        context = "\n\n".join(state["retrieved_docs"])
        state["answer"] = llm_answer_with_context(state["query"], context)
        state["path_taken"].append("generate_answer")
        return state

    return generate_answer_node


def honest_fallback_node(state: RAGState) -> RAGState:
    state["answer"] = (
        "I don't have reliable information to answer this confidently. "
        "Could you rephrase, or point me to the right source?"
    )
    state["path_taken"].append("honest_fallback")
    return state


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.7


def route_after_router(state: RAGState) -> str:
    return "retrieve" if state["needs_retrieval"] else "direct_answer"


def route_after_confidence(state: RAGState) -> str:
    if state["confidence_score"] >= CONFIDENCE_THRESHOLD:
        return "generate_answer"
    if state["retry_count"] < state["max_retries"]:
        return "reformulate"
    return "honest_fallback"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agentic_rag_graph(
    llm_decide: Callable[[str], str],
    vector_search: Callable[[str, int], List[str]],
    llm_score: Callable[[str, str], float],
    llm_reformulate: Callable[[str, str], str],
    llm_answer_direct: Callable[[str], str],
    llm_answer_with_context: Callable[[str, str], str],
    retrieval_k: int = 5,
):
    """Wire the full agentic RAG graph from injected model/retriever callables."""

    graph = StateGraph(RAGState)

    graph.add_node("router", make_router_node(llm_decide))
    graph.add_node("retrieve", make_retrieve_node(vector_search, k=retrieval_k))
    graph.add_node("confidence_check", make_confidence_check_node(llm_score))
    graph.add_node("reformulate", make_reformulate_node(llm_reformulate))
    graph.add_node("direct_answer", make_direct_answer_node(llm_answer_direct))
    graph.add_node("generate_answer", make_generate_answer_node(llm_answer_with_context))
    graph.add_node("honest_fallback", honest_fallback_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router)
    graph.add_edge("retrieve", "confidence_check")
    graph.add_conditional_edges("confidence_check", route_after_confidence)
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate_answer", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("honest_fallback", END)

    return graph.compile()


def initial_state(query: str, max_retries: int = 2) -> RAGState:
    return RAGState(
        query=query,
        original_query=query,
        needs_retrieval=False,
        retrieved_docs=[],
        confidence_score=0.0,
        retry_count=0,
        max_retries=max_retries,
        answer=None,
        path_taken=[],
    )
