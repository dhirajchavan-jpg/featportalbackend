import pytest
from unittest.mock import patch, MagicMock
from app.core.llm_provider import RemoteDenseEmbedder
from app.core.llm_provider import RemoteReranker


def test_embed_documents_success():
    """
    Summary:
        Verify that document embedding succeeds when the remote embedding service responds correctly.

    Explanation:
        This test ensures that the RemoteDenseEmbedder correctly calls the remote
        embedding endpoint, parses the JSON response, and returns the embeddings
        exactly as provided by the service.
    """
    embedder = RemoteDenseEmbedder("http://fake/embed")

    mock_response = MagicMock()
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        result = embedder.embed_documents(["hello"])

    assert result == [[0.1, 0.2]]


def test_embed_documents_failure_returns_zero_vectors():
    """
    Summary:
        Embedding failures must fallback to zero vectors to maintain pipeline stability.

    Explanation:
        When the remote embedding service fails or becomes unavailable, the system
        should not crash. Instead, it must return fixed-size zero vectors for each
        input document so downstream retrieval logic can continue safely.
    """
    embedder = RemoteDenseEmbedder("http://fake/embed")

    with patch("httpx.Client.post", side_effect=Exception("boom")):
        result = embedder.embed_documents(["hello", "world"])

    assert len(result) == 2
    assert all(len(v) == 1024 for v in result)




def test_rerank_fallback_on_failure():
    """
    Summary:
        Reranking failures must fallback to the original document order.

    Explanation:
        If the remote reranking service is unavailable or errors out, the system
        should gracefully degrade by returning the top-k documents from the
        original list instead of blocking the response.
    """
    reranker = RemoteReranker("http://fake/rerank")

    docs = [{"id": 1}, {"id": 2}, {"id": 3}]

    with patch("httpx.Client.post", side_effect=Exception("down")):
        result = reranker.rerank("query", docs, top_k=2)

    assert result == docs[:2]
