"""
RAG pipeline: embeddings, FAISS vector store, and retrieval.
"""

import logging
from typing import List, Tuple

import numpy as np
import faiss
from openai import OpenAI

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_INPUT_CHARS = 8_000  # ~2k tokens — safe ceiling for embedding calls


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_batch(texts: List[str], client: OpenAI) -> np.ndarray:
    """
    Call the OpenAI embeddings endpoint for a list of texts.
    Truncates each text to MAX_INPUT_CHARS to stay within token limits.
    Returns a float32 NumPy array of shape (len(texts), EMBEDDING_DIM).
    """
    safe_texts = [t[:MAX_INPUT_CHARS] for t in texts]
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=safe_texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def build_vector_store(
    chunks: List[str], client: OpenAI
) -> Tuple[faiss.Index, List[str]]:
    """
    Embed *chunks* and load them into a FAISS inner-product index
    (equivalent to cosine similarity after L2 normalisation).

    Returns:
        (index, chunks) — the populated FAISS index and the original chunk list
        so callers can map indices back to text.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from an empty chunk list.")

    logger.info("Embedding %d chunks …", len(chunks))
    embeddings = _embed_batch(chunks, client)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    logger.info("Vector store ready (%d vectors).", index.ntotal)
    return index, chunks


def retrieve_relevant_chunks(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    client: OpenAI,
    k: int = 5,
) -> List[str]:
    """
    Embed *query* and return the top-k most similar chunks.
    """
    k = min(k, len(chunks))
    query_vec = _embed_batch([query], client)
    faiss.normalize_L2(query_vec)

    _scores, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def build_context(chunks: List[str], max_words: int = 2_500) -> str:
    """
    Concatenate *chunks* into a single context string without exceeding
    *max_words* words (a rough proxy for the token budget).
    """
    context_parts: List[str] = []
    total = 0
    for chunk in chunks:
        word_count = len(chunk.split())
        if total + word_count > max_words:
            break
        context_parts.append(chunk)
        total += word_count
    return "\n\n".join(context_parts).strip()
