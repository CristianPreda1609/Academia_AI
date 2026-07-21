"""
Unit tests for EmbeddingsClient.

No real network calls are made: Ollama and Azure are external services and are
out of scope for unit tests. `requests.post` is replaced with a fake so the
error handling in get_embedding() can be exercised deterministically.
"""

import json
import logging

import pytest
import requests

import embeddings_client
from embeddings_client import EmbeddingsClient


client = EmbeddingsClient()


# --- cosine_similarity ------------------------------------------------------

def test_identical_vectors():
    """Identical vectors point the same way -> similarity 1.0."""
    assert client.cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_opposite_vectors():
    """Opposite vectors point the other way -> similarity -1.0."""
    assert client.cosine_similarity([1, 2, 3], [-1, -2, -3]) == pytest.approx(-1.0)


def test_orthogonal_vectors():
    """Perpendicular vectors share no direction -> similarity 0.0."""
    assert client.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_different_vectors():
    """Different but related vectors land strictly between 0 and 1."""
    similarity = client.cosine_similarity([1, 2, 3], [4, 5, 6])
    assert similarity == pytest.approx(0.9746318, abs=1e-6)
    assert 0 < similarity < 1


def test_zero_first_vector():
    """Covers the `magnitude1 == 0` branch."""
    assert client.cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_zero_second_vector():
    """Covers the `magnitude2 == 0` branch."""
    assert client.cosine_similarity([1, 2, 3], [0, 0, 0]) == 0.0


# --- _headers ---------------------------------------------------------------

def test_headers_use_api_key_on_azure(monkeypatch):
    """Azure authenticates with the `api-key` header, not a bearer token."""
    monkeypatch.setattr(
        embeddings_client, "EMBEDDINGS_ENDPOINT",
        "https://foundry.openai.azure.com/openai/v1/embeddings"
    )
    monkeypatch.setattr(embeddings_client, "API_KEY", "secret")

    headers = client._headers()

    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_headers_use_bearer_token_elsewhere(monkeypatch):
    """Any non-Azure endpoint (Ollama, OpenAI, ...) gets a bearer token."""
    monkeypatch.setattr(
        embeddings_client, "EMBEDDINGS_ENDPOINT", "http://localhost:11434/api/embed"
    )
    monkeypatch.setattr(embeddings_client, "API_KEY", "secret")

    headers = client._headers()

    assert headers["Authorization"] == "Bearer secret"
    assert "api-key" not in headers


# --- get_embedding ----------------------------------------------------------

class FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._payload


def test_get_embedding_returns_first_vector(monkeypatch):
    """The API wraps the vector in a list; we unwrap it."""
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse({"embeddings": [[0.1, 0.2, 0.3]]})

    monkeypatch.setattr(embeddings_client.requests, "post", fake_post)

    assert client.get_embedding("salut") == [0.1, 0.2, 0.3]
    assert captured["json"]["input"] == "salut"
    assert captured["json"]["model"] == embeddings_client.EMBEDDINGS_MODEL
    assert captured["url"] == embeddings_client.EMBEDDINGS_ENDPOINT
    assert captured["timeout"] == 60


@pytest.mark.parametrize(
    "raised, expected_message",
    [
        (requests.exceptions.ConnectionError(), "Is Ollama running?"),
        (requests.exceptions.Timeout(), "did not answer in time"),
    ],
)
def test_get_embedding_wraps_transport_errors(monkeypatch, raised, expected_message):
    """Transport failures become a ConnectionError with an actionable message."""
    def fake_post(*args, **kwargs):
        raise raised

    monkeypatch.setattr(embeddings_client.requests, "post", fake_post)

    with pytest.raises(ConnectionError, match=expected_message):
        client.get_embedding("salut")


def test_get_embedding_wraps_http_errors(monkeypatch):
    """A 4xx/5xx answer is reported with its status code and body."""
    monkeypatch.setattr(
        embeddings_client.requests, "post",
        lambda *a, **kw: FakeResponse(status_code=500, text="boom")
    )

    with pytest.raises(ConnectionError, match=r"HTTP 500 - boom"):
        client.get_embedding("salut")


# --- semantic_search --------------------------------------------------------

def write_embeddings_file(tmp_path, monkeypatch, content):
    """Points EMBEDDINGS_FILE at a temporary file holding `content`."""
    path = tmp_path / "embeddings.json"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(embeddings_client, "EMBEDDINGS_FILE", str(path))
    return path


def stub_embedding(monkeypatch, vector):
    """Makes get_embedding() return `vector` without touching the network."""
    monkeypatch.setattr(
        EmbeddingsClient, "get_embedding", lambda self, text: vector
    )


def chunk(document_id, index, embedding):
    return {
        "document_id": document_id,
        "chunk_index": index,
        "embedding": embedding,
        "content": f"{document_id} #{index}",
    }


def test_semantic_search_sorts_and_filters(tmp_path, monkeypatch):
    """Chunks come back best-first, and anything at or below the threshold is dropped."""
    stub_embedding(monkeypatch, [1, 0])
    write_embeddings_file(tmp_path, monkeypatch, json.dumps([
        chunk("weak", 0, [1, 3]),      # ~0.32 -> filtered out
        chunk("perfect", 0, [1, 0]),   # 1.00
        chunk("unrelated", 0, [0, 1]),  # 0.00 -> filtered out
        chunk("decent", 0, [1, 1]),    # ~0.71
    ]))
    monkeypatch.setattr(embeddings_client, "SIMILARITY_THRESHOLD", 0.5)
    monkeypatch.setattr(embeddings_client, "TOP_N", 4)

    results = client.semantic_search("intrebare")

    assert [r["document_id"] for r in results] == ["perfect", "decent"]
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[0]["content"] == "perfect #0"
    assert results[0]["chunk_index"] == 0
    assert "embedding" not in results[0]


def test_semantic_search_respects_top_n(tmp_path, monkeypatch):
    """Only the TOP_N best chunks are returned, even if more pass the threshold."""
    stub_embedding(monkeypatch, [1, 0])
    write_embeddings_file(tmp_path, monkeypatch, json.dumps([
        chunk("a", 0, [1, 0]),
        chunk("b", 0, [1, 1]),
        chunk("c", 0, [2, 1]),
    ]))
    monkeypatch.setattr(embeddings_client, "SIMILARITY_THRESHOLD", 0.5)
    monkeypatch.setattr(embeddings_client, "TOP_N", 1)

    results = client.semantic_search("intrebare")

    assert [r["document_id"] for r in results] == ["a"]


def test_semantic_search_returns_empty_when_nothing_passes(tmp_path, monkeypatch):
    """No chunk above the threshold is not an error - just no context."""
    stub_embedding(monkeypatch, [1, 0])
    write_embeddings_file(tmp_path, monkeypatch, json.dumps([chunk("a", 0, [0, 1])]))
    monkeypatch.setattr(embeddings_client, "SIMILARITY_THRESHOLD", 0.5)

    assert client.semantic_search("intrebare") == []


def test_semantic_search_survives_embedding_failure(monkeypatch, caplog):
    """If the embeddings server is down, search degrades instead of crashing."""
    def boom(self, text):
        raise ConnectionError("server down")

    monkeypatch.setattr(EmbeddingsClient, "get_embedding", boom)

    with caplog.at_level(logging.ERROR):
        assert client.semantic_search("intrebare") == []
    assert "server down" in caplog.text


def test_semantic_search_survives_missing_file(tmp_path, monkeypatch, caplog):
    """A missing embeddings.json is logged and skipped, not raised."""
    stub_embedding(monkeypatch, [1, 0])
    monkeypatch.setattr(
        embeddings_client, "EMBEDDINGS_FILE", str(tmp_path / "nope.json")
    )

    with caplog.at_level(logging.ERROR):
        assert client.semantic_search("intrebare") == []
    assert "does not exist" in caplog.text


def test_semantic_search_survives_corrupted_file(tmp_path, monkeypatch, caplog):
    """A truncated or hand-edited embeddings.json is logged and skipped."""
    stub_embedding(monkeypatch, [1, 0])
    write_embeddings_file(tmp_path, monkeypatch, "{not json")

    with caplog.at_level(logging.ERROR):
        assert client.semantic_search("intrebare") == []
    assert "corrupted" in caplog.text
