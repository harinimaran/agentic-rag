"""Hybrid retrieval: BM25 + dense recall, fused by RRF, then cross-encoder re-ranking."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


class HybridRetriever:
    def __init__(self, index_dir: Path, rerank: bool = True):
        self.chunks = json.loads((index_dir / "chunks.json").read_text())
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.bm25 = pickle.loads((index_dir / "bm25.pkl").read_bytes())
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL) if rerank else None

    def _dense(self, query: str, k: int) -> list[int]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        _, idx = self.index.search(q, k)
        return idx[0].tolist()

    def _sparse(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(query.lower().split())
        return np.argsort(scores)[::-1][:k].tolist()

    @staticmethod
    def _rrf(*rankings: list[int]) -> list[int]:
        """Reciprocal rank fusion — combines rankings without score normalisation."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])]

    def search(self, query: str, k: int = 5, pool: int = 30) -> list[dict]:
        candidates = self._rrf(self._dense(query, pool), self._sparse(query, pool))[:pool]
        hits = [self.chunks[i] for i in candidates]

        if self.reranker is not None and hits:
            pairs = [(query, h["text"]) for h in hits]
            scores = self.reranker.predict(pairs)
            order = np.argsort(scores)[::-1]
            hits = [hits[i] | {"score": float(scores[i])} for i in order]
        else:
            hits = [h | {"score": 0.0} for h in hits]

        return hits[:k]
