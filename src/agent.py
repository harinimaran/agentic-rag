"""LangGraph agent: decompose -> retrieve -> grade -> (retry | answer).

The graph is the point. A single retrieve-then-generate call fails on questions
that need two different facts; decomposition plus a grading gate fixes most of it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from .retrieve import HybridRetriever

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_ATTEMPTS = 2
client = OpenAI()


class State(TypedDict):
    question: str
    subqueries: list[str]
    context: Annotated[list[dict], lambda a, b: a + b]
    grade: float
    attempts: int
    answer: str


def _chat(system: str, user: str, as_json: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        **kwargs,
    )
    return resp.choices[0].message.content


def build_graph(retriever: HybridRetriever):
    def decompose(state: State) -> dict:
        out = _chat(
            "Split the question into 1-3 standalone search queries. "
            'Return JSON: {"queries": [...]}. One query if it is already atomic.',
            state["question"],
            as_json=True,
        )
        queries = json.loads(out).get("queries") or [state["question"]]
        return {"subqueries": queries[:3], "attempts": state.get("attempts", 0)}

    def retrieve(state: State) -> dict:
        seen, hits = set(), []
        k = 5 if state["attempts"] == 0 else 8  # widen on retry
        for q in state["subqueries"]:
            for h in retriever.search(q, k=k):
                key = (h["doc_id"], h["chunk_id"])
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
        return {"context": hits}

    def grade(state: State) -> dict:
        ctx = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(state["context"]))
        out = _chat(
            "Rate 0.0-1.0 how fully the context answers the question. "
            'Return JSON: {"score": float}.',
            f"Question: {state['question']}\n\nContext:\n{ctx}",
            as_json=True,
        )
        return {"grade": float(json.loads(out)["score"]),
                "attempts": state["attempts"] + 1}

    def answer(state: State) -> dict:
        ctx = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(state["context"]))
        out = _chat(
            "Answer using ONLY the numbered context. Cite sources inline as [i]. "
            "If the context does not contain the answer, say so plainly. "
            "Never add facts that are not in the context.",
            f"Question: {state['question']}\n\nContext:\n{ctx}",
        )
        return {"answer": out}

    def route(state: State) -> str:
        if state["grade"] >= 0.6 or state["attempts"] >= MAX_ATTEMPTS:
            return "answer"
        return "retrieve"

    g = StateGraph(State)
    for name, fn in [("decompose", decompose), ("retrieve", retrieve),
                     ("grade", grade), ("answer", answer)]:
        g.add_node(name, fn)
    g.set_entry_point("decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


def ask(question: str, index_dir: Path = Path("index")) -> dict:
    graph = build_graph(HybridRetriever(index_dir))
    final = graph.invoke({"question": question, "context": [], "attempts": 0})
    return {
        "answer": final["answer"],
        "grade": final["grade"],
        "attempts": final["attempts"],
        "sources": [{"doc_id": c["doc_id"], "title": c["title"]} for c in final["context"]],
    }
