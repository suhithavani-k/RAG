import numpy as np
import pytest

from core.chunker import Chunk
from core.embeddings import encode_texts
from core.rag_pipeline import NOT_FOUND, answer_question
from core.retriever import cosine_scores, retrieve


class FakeEncoder:
    def encode(self, sentences, **kwargs):
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class NeverCallClient:
    def chat(self, *args, **kwargs):
        raise AssertionError("Generation must not run when retrieval finds no relevant chunks")


class RecordingClient:
    def __init__(self):
        self.call = None

    def chat(self, model, system_prompt, user_prompt, temperature):
        self.call = (model, system_prompt, user_prompt, temperature)
        return "The application requires a completed application."


def test_embedding_generation_returns_matrix():
    vectors = encode_texts(FakeEncoder(), ["query", "document"])
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_cosine_similarity_and_top_k_order():
    chunks = [Chunk("a", "guide.pdf", 1, "eligibility requirements"), Chunk("b", "guide.pdf", 2, "registration dates")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = cosine_scores(np.array([1.0, 0.0]), vectors)
    assert np.allclose(scores, [1.0, 0.0])
    results = retrieve(np.array([1.0, 0.0]), chunks, vectors, top_k=1, min_similarity=0.2)
    assert results[0].chunk.chunk_id == "a"
    assert results[0].score == pytest.approx(1.0)


def test_threshold_prevents_irrelevant_generation():
    chunks = [Chunk("a", "guide.pdf", 1, "eligibility requirements")]
    result, sources = answer_question(
        "Tell me about eligibility", chunks, np.array([[0.0, 1.0]]), np.array([1.0, 0.0]),
        NeverCallClient(), "qwen3:4b", min_similarity=0.5,
    )
    assert result == NOT_FOUND
    assert sources == []


def test_generation_receives_question_and_retrieved_source_context_only():
    chunks = [Chunk("a", "guide.pdf", 2, "Eligibility requires a completed application.")]
    client = RecordingClient()
    answer, sources = answer_question(
        "What is required?", chunks, np.array([[1.0, 0.0]]), np.array([1.0, 0.0]),
        client, "qwen3:4b", min_similarity=0.2,
    )
    assert answer.startswith("The application requires")
    assert len(sources) == 1
    assert client.call[0] == "qwen3:4b"
    assert "What is required?" in client.call[2]
    assert "guide.pdf, page 2" in client.call[2]
    assert "Eligibility requires a completed application." in client.call[2]
    assert "untrusted reference data" in client.call[1]


def test_invalid_embedding_dimensions_fail():
    with pytest.raises(ValueError, match="matching dimensions"):
        cosine_scores(np.array([1.0]), np.array([[1.0, 0.0]]))
