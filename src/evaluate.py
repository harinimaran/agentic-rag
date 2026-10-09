"""Evaluation harness: faithfulness, context precision, answer relevance.

Implemented directly rather than via a library so each metric is inspectable —
the numbers on the resume come from this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .agent import _chat, ask

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def faithfulness(answer: str, context: str) -> float:
    """Fraction of claims in the answer that the context actually supports."""
    out = _chat(
        "Extract every factual claim from the answer. For each, say whether the "
        'context supports it. Return JSON: {"claims": [{"claim": str, "supported": bool}]}.',
        f"Answer: {answer}\n\nContext:\n{context}",
        as_json=True,
    )
    claims = json.loads(out)["claims"]
    return sum(c["supported"] for c in claims) / len(claims) if claims else 0.0


def context_precision(contexts: list[str], ground_truth: str, embedder) -> float:
    """Share of retrieved chunks semantically close to the reference answer."""
    if not contexts:
        return 0.0
    vecs = embedder.encode([ground_truth] + contexts, normalize_embeddings=True)
    sims = vecs[1:] @ vecs[0]
    return float((sims > 0.4).mean())


def answer_relevance(question: str, answer: str, embedder) -> float:
    vecs = embedder.encode([question, answer], normalize_embeddings=True)
    return float(vecs[0] @ vecs[1])


def run(eval_path: Path, index_dir: Path, out_path: Path) -> None:
    embedder = SentenceTransformer(EMBED_MODEL)
    rows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    results = []

    for i, row in enumerate(rows, 1):
        res = ask(row["question"], index_dir)
        ctx_texts = [s.get("text", "") for s in res["sources"]]
        ctx_blob = "\n\n".join(ctx_texts)
        results.append({
            "question": row["question"],
            "faithfulness": faithfulness(res["answer"], ctx_blob),
            "context_precision": context_precision(ctx_texts, row["ground_truth"], embedder),
            "answer_relevance": answer_relevance(row["question"], res["answer"], embedder),
            "attempts": res["attempts"],
        })
        print(f"{i}/{len(rows)}", end="\r")

    summary = {
        m: round(float(np.mean([r[m] for r in results])), 3)
        for m in ("faithfulness", "context_precision", "answer_relevance")
    }
    out_path.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print("\n", summary)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=Path, required=True, help="JSONL: question, ground_truth")
    p.add_argument("--index", type=Path, default=Path("index"))
    p.add_argument("--out", type=Path, default=Path("eval_results.json"))
    a = p.parse_args()
    run(a.eval, a.index, a.out)
