import pytest

from app.services.retrieval.rrf_fusion import (
    RRFFusion,
    get_rrf_fusion
)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def rrf():
    # Use small k for easier math verification
    return RRFFusion(k=10)


@pytest.fixture
def bm25_results():
    return [
        {
            "id": "doc1",
            "content": "bm25 content 1",
            "metadata": {"source": "a"},
            "score": 10.0,
        },
        {
            "id": "doc2",
            "content": "bm25 content 2",
            "metadata": {"source": "a"},
            "score": 8.0,
        },
    ]


@pytest.fixture
def vector_results():
    return [
        {
            "id": "doc2",
            "content": "vector content 2",
            "metadata": {"source": "b"},
            "score": 0.92,
        },
        {
            "id": "doc3",
            "content": "vector content 3",
            "metadata": {"source": "b"},
            "score": 0.85,
        },
    ]


# -------------------------
# fuse()
# -------------------------

def test_rrf_fusion_combines_and_deduplicates(rrf, bm25_results, vector_results):
    results = rrf.fuse(bm25_results, vector_results, top_k=10)

    ids = [r["id"] for r in results]

    # Deduplicated and combined
    assert set(ids) == {"doc1", "doc2", "doc3"}
    assert len(results) == 3


def test_rrf_fusion_rank_order(rrf, bm25_results, vector_results):
    results = rrf.fuse(bm25_results, vector_results, top_k=10)

    # doc2 appears in BOTH → should rank highest
    assert results[0]["id"] == "doc2"
    assert results[0]["found_in"] == ["bm25", "vector"]
    assert results[0]["rrf_rank"] == 1


def test_rrf_fusion_found_in_flags(rrf, bm25_results, vector_results):
    results = rrf.fuse(bm25_results, vector_results)

    doc1 = next(r for r in results if r["id"] == "doc1")
    doc3 = next(r for r in results if r["id"] == "doc3")

    assert doc1["found_in"] == ["bm25"]
    assert doc3["found_in"] == ["vector"]


def test_rrf_fusion_top_k_limit(rrf, bm25_results, vector_results):
    results = rrf.fuse(bm25_results, vector_results, top_k=2)
    assert len(results) == 2


def test_rrf_fusion_empty_inputs(rrf):
    results = rrf.fuse([], [])
    assert results == []


# -------------------------
# get_fusion_stats()
# -------------------------

def test_get_fusion_stats(rrf, bm25_results, vector_results):
    results = rrf.fuse(bm25_results, vector_results)

    stats = rrf.get_fusion_stats(results)

    assert stats["total_results"] == 3
    assert stats["found_in_both"] == 1
    assert stats["found_in_bm25_only"] == 1
    assert stats["found_in_vector_only"] == 1
    assert stats["max_rrf_score"] >= stats["min_rrf_score"]


def test_get_fusion_stats_empty(rrf):
    stats = rrf.get_fusion_stats([])
    assert stats == {}


# -------------------------
# explain_ranking()
# -------------------------

def test_explain_ranking_both_methods(rrf, bm25_results, vector_results):
    results = rrf.fuse(bm25_results, vector_results)
    doc = results[0]

    explanation = rrf.explain_ranking(doc)

    assert "RRF Rank" in explanation
    assert "RRF Score" in explanation
    assert "BM25 Rank" in explanation
    assert "Vector Rank" in explanation
    assert "both retrieval methods" in explanation


def test_explain_ranking_bm25_only(rrf, bm25_results):
    results = rrf.fuse(bm25_results, [])
    doc = results[0]

    explanation = rrf.explain_ranking(doc)
    assert "keyword search only" in explanation


def test_explain_ranking_vector_only(rrf, vector_results):
    results = rrf.fuse([], vector_results)
    doc = results[0]

    explanation = rrf.explain_ranking(doc)
    assert "semantic search only" in explanation


# -------------------------
# Singleton
# -------------------------

def test_get_rrf_fusion_singleton():
    r1 = get_rrf_fusion()
    r2 = get_rrf_fusion()
    assert r1 is r2
