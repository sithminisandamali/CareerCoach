"""
ingest.py
Builds a FAISS vector index from the domain corpus in data/corpus/*.txt

Chunking strategy: fixed-size character chunks (~800 chars) with 100-char
overlap. Simple, deterministic, and easy to explain/justify in the README
and the viva ("why 800/100?" -> keeps each chunk to roughly one Q&A item
or one guideline paragraph from our corpus, overlap avoids cutting a
sentence in half at chunk boundaries).

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - runs locally, free, no API key needed, small enough for Streamlit Cloud.

Vector store: FAISS (in-memory, saved to disk as index.faiss + chunks.pkl)
"""

import os
import pickle
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.faiss")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.pkl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def chunk_text(text: str, filename: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": filename})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index():
    all_chunks = []
    for filepath in glob.glob(os.path.join(CORPUS_DIR, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, os.path.basename(filepath)))

    if not all_chunks:
        raise RuntimeError(f"No corpus files found in {CORPUS_DIR}. Add .txt files first.")

    embedder = get_embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Indexed {len(all_chunks)} chunks from {len(glob.glob(os.path.join(CORPUS_DIR, '*.txt')))} files.")


if __name__ == "__main__":
    build_index()
