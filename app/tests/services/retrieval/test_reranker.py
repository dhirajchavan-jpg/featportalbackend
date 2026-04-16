import pytest
from unittest.mock import patch, MagicMock
import requests

from app.services.retrieval.reranker import (
    BGEReranker,
    get_reranker
)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def reranker():
    return BGEReranker()


@pytest.fixture
def sample_docs():
    return [
        {"id": "1", "content": "Document one"},
        {"id": "2", "content": "Document two"},
        {"id": "3", "content": "Document three"},
    ]


@pytest.fixture
def reranked_response():
    return {
        "results": [
            {"id": "2", "content": "Document two", "rerank_score": 0.95},
            {"id": "1", "content": "Document one", "rerank_score": 0.85},
        ]
    }


# -------------------------
# Core rerank() tests
# -------------------------

def test_rerank_empty_documents(reranker):
    results = reranker.rerank("test query", [])
    assert results == []


@patch("app.services.retrieval.reranker.requests.post")
def test_rerank_success(mock_post, reranker, sample_docs, reranked_response):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = reranked_response
    mock_post.return_value = mock_response

    results = reranker.rerank("test query", sample_docs, top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "2"
    assert results[0]["final_rank"] == 1
    assert results[1]["final_rank"] == 2


@patch("app.services.retrieval.reranker.requests.post")
def test_rerank_server_error_fallback(mock_post, reranker, sample_docs):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    results = reranker.rerank("test query", sample_docs, top_k=2)

    # Fallback should return original order
    assert results == sample_docs[:2]


@patch("app.services.retrieval.reranker.requests.post")
def test_rerank_connection_error_fallback(mock_post, reranker, sample_docs):
    mock_post.side_effect = requests.exceptions.ConnectionError

    results = reranker.rerank("test query", sample_docs, top_k=2)

    assert results == sample_docs[:2]


@patch("app.services.retrieval.reranker.requests.post")
def test_rerank_generic_exception_fallback(mock_post, reranker, sample_docs):
    mock_post.side_effect = Exception("Unexpected failure")

    results = reranker.rerank("test query", sample_docs, top_k=2)

    assert results == sample_docs[:2]


# -------------------------
# batch_rerank
# -------------------------

@patch.object(BGEReranker, "rerank")
def test_batch_rerank_delegates(mock_rerank, reranker, sample_docs):
    mock_rerank.return_value = sample_docs[:1]

    results = reranker.batch_rerank("query", sample_docs, top_k=1)

    mock_rerank.assert_called_once()
    assert results == sample_docs[:1]


# -------------------------
# get_reranking_stats
# -------------------------

def test_get_reranking_stats(reranker):
    docs = [
        {"rerank_score": 0.9},
        {"rerank_score": 0.7},
        {"rerank_score": 0.8},
    ]

    stats = reranker.get_reranking_stats(docs)

    assert stats["total_reranked"] == 3
    assert stats["avg_rerank_score"] == pytest.approx(0.8)
    assert stats["max_rerank_score"] == 0.9
    assert stats["min_rerank_score"] == 0.7


def test_get_reranking_stats_empty(reranker):
    assert reranker.get_reranking_stats([]) == {}


# -------------------------
# explain_reranking
# -------------------------

def test_explain_reranking_full(reranker):
    doc = {
        "final_rank": 1,
        "rerank_score": 0.92,
        "rrf_rank": 3,
    }

    explanation = reranker.explain_reranking(doc)

    assert "Final Rank: 1" in explanation
    assert "Rerank Score: 0.9200" in explanation
    assert "Moved UP 2 positions" in explanation


def test_explain_reranking_no_info(reranker):
    explanation = reranker.explain_reranking({})
    assert explanation == "No reranking info"


# -------------------------
# Singleton behavior
# -------------------------

def test_get_reranker_singleton():
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2
