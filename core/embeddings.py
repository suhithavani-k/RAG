from typing import Protocol

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    allow_download: bool = False,
) -> TextEncoder:
    """Load the compact CPU model, requiring explicit permission for a first download."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu", local_files_only=not allow_download)


def encode_texts(model: TextEncoder, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = np.asarray(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("The embedding model returned an invalid vector array.")
    return vectors
