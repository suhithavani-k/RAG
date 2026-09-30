from dataclasses import dataclass
import re

import numpy as np

from core.chunker import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def cosine_scores(query_vector: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for rows, safely handling zero-length vectors."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    vectors = np.asarray(chunk_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != query.shape[0]:
        raise ValueError("Query and document embeddings must have matching dimensions.")
    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denominator = vector_norms * query_norm
    scores = np.zeros(vectors.shape[0], dtype=np.float32)
    valid = denominator > 0
    scores[valid] = (vectors[valid] @ query) / denominator[valid]
    return np.clip(scores, -1.0, 1.0)


def _is_redundant(candidate: str, selected: list[str]) -> bool:
    candidate_words = set(re.findall(r"\w+", candidate.lower()))
    if not candidate_words:
        return True
    for text in selected:
        words = set(re.findall(r"\w+", text.lower()))
        if words and len(candidate_words & words) / min(len(candidate_words), len(words)) >= 0.82:
            return True
    return False


def retrieve(
    query_vector: np.ndarray,
    chunks: list[Chunk],
    chunk_vectors: np.ndarray,
    top_k: int = 4,
    min_similarity: float | None = 0.20,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise ValueError("Top-K must be at least 1.")
    if len(chunks) != len(chunk_vectors):
        raise ValueError("Every chunk must have exactly one embedding.")
    if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
        raise ValueError("Similarity threshold must be between -1 and 1.")
    scores = cosine_scores(query_vector, chunk_vectors)
    order = np.argsort(-scores, kind="stable")
    results: list[RetrievedChunk] = []
    selected_text: list[str] = []
    for index in order:
        score = float(scores[index])
        if min_similarity is not None and score < min_similarity:
            continue
        chunk = chunks[int(index)]
        if _is_redundant(chunk.text, selected_text):
            continue
        results.append(RetrievedChunk(chunk, score))
        selected_text.append(chunk.text)
        if len(results) >= top_k:
            break
    return results
