# Agentic RAG Assistant with Evaluation Harness

Retrieval agent that decomposes a question, retrieves hybrid (BM25 + dense), grades
its own context, and retries before answering. Ships with an evaluation harness so
every quality claim is reproducible.

## Why it is built this way

A single retrieve-then-generate call fails on two common cases: questions needing
two unrelated facts, and questions where the first retrieval misses. The graph
handles both — decomposition for the first, a grading gate with a widened retry
for the second.

```
decompose ──▶ retrieve ──▶ grade ──┬─(score ≥ 0.6 or 2 attempts)─▶ answer ──▶ END
                  ▲                │
                  └────────────────┘  (retry, wider k)
```

## Results

Measured on a 200-question labelled set over the ingested corpus.

| Configuration | Faithfulness | Context precision | Answer relevance |
|---|---|---|---|
| Dense only, no agent | 0.68 | 0.54 | 0.81 |
| + hybrid BM25 + RRF | 0.79 | 0.66 | 0.83 |
| + cross-encoder re-rank | 0.86 | 0.74 | 0.85 |
| + decompose & grade loop | **0.91** | **0.78** | **0.87** |

Regenerate with `python -m src.evaluate --eval data/eval.jsonl`.
Numbers above are from the committed `eval_results.json`.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

Corpus format — one JSON object per line:

```json
{"id": "doc-001", "title": "Governor Limits", "text": "Full document text..."}
```

Eval set format:

```json
{"question": "What is the SOQL query row limit?", "ground_truth": "50,000 rows per transaction."}
```

## Run

```bash
python -m src.ingest --corpus data/corpus.jsonl --out index
python -m src.evaluate --eval data/eval.jsonl --index index
uvicorn app:app --reload
```

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How do I avoid hitting the DML row limit in a batch job?"}'
```

Docker:

```bash
docker build -t agentic-rag . && docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY agentic-rag
```

## Layout

```
src/ingest.py     chunking, embedding, FAISS + BM25 index build
src/retrieve.py   hybrid retrieval, RRF fusion, cross-encoder re-ranking
src/agent.py      LangGraph state machine
src/evaluate.py   faithfulness / context precision / answer relevance
app.py            FastAPI endpoint with latency + grade logging
```

## Notes and limitations

- Faithfulness uses an LLM judge, so it carries that model's bias. The relative
  improvement across configurations is more trustworthy than the absolute value.
- `IndexFlatIP` is exact search. Past ~1M chunks, switch to `IndexIVFFlat`.
- The retry widens `k` but reuses the same queries; rewriting the query on retry
  would likely help more.
