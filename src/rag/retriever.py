"""
retriever.py
Loads the FAISS index built by ingest.py and retrieves top-k relevant chunks
for a given query. Also exposes a rerank() function that uses a Groq model
to re-score the top candidates (demonstrates the "retrieval re-ranking"
sub-task from the model-selection table).
"""

import os
import pickle
import faiss
from .ingest import get_embedder, INDEX_PATH, CHUNKS_PATH
from ..models.llm_clients import call_groq, MODEL_RERANK

_index = None
_chunks = None


def _load():
    global _index, _chunks
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            # Auto-build on first run (e.g. first request on Streamlit Cloud)
            from .ingest import build_index
            build_index()
        _index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)
    return _index, _chunks


def retrieve(query: str, k: int = 5):
    index, chunks = _load()
    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


def rerank(query: str, candidates: list, top_n: int = 3):
    """Use a mid-tier Groq model to re-score retrieved chunks by relevance.
    Demonstrates deliberate model selection for the 'retrieval re-ranking'
    sub-task (fast enough to score several chunks without becoming a
    bottleneck, per the model comparison table in the README)."""
    if not candidates:
        return []

    numbered = "\n".join(f"[{i}] {c['text'][:300]}" for i, c in enumerate(candidates))
    prompt = (
        f"Query: {query}\n\nCandidate passages:\n{numbered}\n\n"
        f"Return ONLY a comma-separated list of the passage numbers, "
        f"ordered from most to least relevant to the query."
    )
    messages = [{"role": "user", "content": prompt}]
    response = call_groq(MODEL_RERANK, messages, temperature=0.0, max_tokens=50)

    try:
        order = [int(x.strip()) for x in response.strip().split(",") if x.strip().isdigit()]
        ranked = [candidates[i] for i in order if i < len(candidates)]
        # append any not mentioned, keep original order
        seen = set(order)
        ranked += [c for i, c in enumerate(candidates) if i not in seen]
        return ranked[:top_n]
    except Exception:
        # fall back to original FAISS order if the reranker output is malformed
        return candidates[:top_n]
