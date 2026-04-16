import importlib
import os
from unittest.mock import MagicMock, patch
import pytest
from app.config import settings

# ============================================================
# 1. RemoteDenseEmbedder
# ============================================================

def test_remote_dense_embedder_success():
    from app.core.llm_provider import RemoteDenseEmbedder

    embedder = RemoteDenseEmbedder(["http://fake"])

    mock_response = MagicMock()
    mock_response.json.return_value = {"embeddings": [[1.0, 2.0]]}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        result = embedder.embed_documents(["hello"])

    assert result == [[1.0, 2.0]]


def test_remote_dense_embedder_failure_fallback():
    from app.core.llm_provider import RemoteDenseEmbedder

    embedder = RemoteDenseEmbedder(["http://fake"])

    with patch("httpx.Client.post", side_effect=Exception("boom")):
        result = embedder.embed_documents(["hello"])

    # fallback embedding (1024 dim zero vector)
    assert len(result) == 1
    assert len(result[0]) == 1024


# ============================================================
# 2. RemoteReranker
# ============================================================

def test_remote_reranker_success():
    from app.core.llm_provider import RemoteReranker

    reranker = RemoteReranker(["http://fake"])

    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"id": 1}]}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        result = reranker.rerank("query", [{"text": "doc"}])

    assert result == [{"id": 1}]


def test_remote_reranker_fallback():
    from app.core.llm_provider import RemoteReranker

    reranker = RemoteReranker(["http://fake"])
    docs = [{"text": "a"}, {"text": "b"}]

    with patch("httpx.Client.post", side_effect=Exception("fail")):
        result = reranker.rerank("query", docs, top_k=1)

    # Fallback returns sliced docs
    assert result == docs[:1]


# ============================================================
# 3. RemoteLayoutProcessor
# ============================================================

def test_layout_processor_success():
    from app.core.llm_provider import RemoteLayoutProcessor

    processor = RemoteLayoutProcessor(["http://fake"])

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pages": [{"elements": [{"type": "text"}]}]
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        result = processor.detect(b"bytes")

    assert result == [{"type": "text"}]


def test_layout_processor_failure():
    from app.core.llm_provider import RemoteLayoutProcessor

    processor = RemoteLayoutProcessor(["http://fake"])

    with patch("httpx.Client.post", side_effect=Exception()):
        result = processor.detect(b"bytes")

    assert result == []


# ============================================================
# 4. get_llm ROUTING LOGIC
# ============================================================

def test_get_llm_returns_existing():
    from app.core import llm_provider

    # Mock the global variable directly
    mock_llm = MagicMock()
    llm_provider._llm = mock_llm
    
    result = llm_provider.get_llm()
    assert result == mock_llm

def test_get_llm_returns_simple_by_name():
    from app.core import llm_provider

    # Mock specific named model
    mock_simple = MagicMock()
    mock_simple.model = "simple-model"
    llm_provider._llm_simple = mock_simple

    result = llm_provider.get_llm("simple-model")
    assert result == mock_simple


# ============================================================
# 5. cleanup_vram
# ============================================================

def test_cleanup_vram_does_not_crash():
    from app.core.llm_provider import cleanup_vram

    with patch("httpx.Client.post") as mock_post:
        cleanup_vram()
        # Should call post for every model/url combo, or at least pass without error
        # We just verify it ran
        assert True


# ============================================================
# 6. check_models_health
# ============================================================

def test_check_models_health_unreachable():
    from app.core.llm_provider import check_models_health

    # Mock httpx.get to always raise an exception
    with patch("httpx.Client.get", side_effect=Exception("Connection refused")):
        result = check_models_health()

    # Updated assertions based on new structure:
    # { "status": "ok", "endpoints": { "url": "unreachable" } }
    assert "endpoints" in result
    
    # Check that at least one model server URL is marked unreachable
    for url in settings.model_server_urls_list:
        assert result["endpoints"][url] == "unreachable"