"""FastAPI wrapper with latency and token logging."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import build_graph
from src.retrieve import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("rag")

app = FastAPI(title="Agentic RAG Assistant")
graph = None


class Query(BaseModel):
    question: str


@app.on_event("startup")
def _load() -> None:
    global graph
    graph = build_graph(HybridRetriever(Path("index")))
    log.info("index loaded")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(q: Query) -> dict:
    t0 = time.perf_counter()
    final = graph.invoke({"question": q.question, "context": [], "attempts": 0})
    elapsed = round((time.perf_counter() - t0) * 1000)
    log.info("q=%r grade=%.2f attempts=%d ms=%d",
             q.question, final["grade"], final["attempts"], elapsed)
    return {
        "answer": final["answer"],
        "retrieval_grade": final["grade"],
        "attempts": final["attempts"],
        "latency_ms": elapsed,
        "sources": [{"doc_id": c["doc_id"], "title": c["title"]}
                    for c in final["context"]],
    }
