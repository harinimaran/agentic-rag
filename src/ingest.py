"""Chunk a document corpus, embed it, and persist a FAISS index + BM25 corpus."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 900
OVERLAP = 150


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    title: str
    text: str


def split(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Sliding-window split that prefers paragraph boundaries."""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            brk = text.rfind("\n\n", start, end)
            if brk > start + size // 2:
                end = brk
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in out if c]


def build(corpus_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []

    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        for i, piece in enumerate(split(doc["text"])):
            chunks.append(Chunk(doc["id"], i, doc.get("title", doc["id"]), piece))

    print(f"{len(chunks)} chunks from {corpus_path}")

    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(
        [c.text for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    bm25 = BM25Okapi([c.text.lower().split() for c in chunks])
    (out_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    (out_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks]))
    print(f"index written to {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True, help="JSONL with id/title/text")
    p.add_argument("--out", type=Path, default=Path("index"))
    a = p.parse_args()
    build(a.corpus, a.out)
