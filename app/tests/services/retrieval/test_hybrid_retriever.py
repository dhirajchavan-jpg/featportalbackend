import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_retrievers():
    with patch(
        "app.services.retrieval.hybrid_retriever.get_bm25_retriever"
    ) as bm25_mock, patch(
        "app.services.retrieval.hybrid_retriever.get_vector_retriever"
    ) as vector_mock, patch(
        "app.services.retrieval.hybrid_retriever.get_reranker"
    ) as reranker_mock:

        bm25 = AsyncMock()
        vector = AsyncMock()
        
        # Reranker has mixed methods: load_model is sync, rerank is async
        reranker = MagicMock()
        reranker.rerank = AsyncMock() # FIX: Explicitly make rerank async

        bm25_mock.return_value = bm25
        vector_mock.return_value = vector
        reranker_mock.return_value = reranker

        yield bm25, vector, reranker


@pytest.fixture
def hybrid(mock_retrievers):
    # This executes __init__ (covers logger + init lines)
    return HybridRetriever()


@pytest.mark.asyncio
async def test_no_sources_returns_empty(hybrid):
    result = await hybrid.retrieve(
        query="test",
        project_id=None,
        selected_sectors=[]
    )

    assert result["results"] == []
    assert result["stats"] == {}


@pytest.mark.asyncio
async def test_vector_only_path(hybrid, mock_retrievers):
    bm25, vector, reranker = mock_retrievers

    vector.retrieve.return_value = [
        {"id": "doc1", "text": "vector"}
    ]

    result = await hybrid.retrieve(
        query="hello",
        project_id="p1",
        selected_sectors=[],
        dense_weight=1.0,
        sparse_weight=0.0,
        use_reranking=False
    )

    assert len(result["results"]) == 1
    vector.retrieve.assert_awaited()
    bm25.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_bm25_only_path(hybrid, mock_retrievers):
    bm25, vector, reranker = mock_retrievers

    bm25.retrieve.return_value = [
        {"id": "doc1", "text": "bm25"}
    ]

    result = await hybrid.retrieve(
        query="hello",
        project_id="p1",
        selected_sectors=["s1"],
        dense_weight=0.0,
        sparse_weight=1.0,
        use_reranking=False
    )

    assert len(result["results"]) == 1
    bm25.retrieve.assert_awaited()
    vector.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_rrf_fusion(hybrid, mock_retrievers):
    bm25, vector, reranker = mock_retrievers

    bm25.retrieve.return_value = [
        {"id": "a", "text": "bm25-a"},
        {"id": "b", "text": "bm25-b"},
    ]

    vector.retrieve.return_value = [
        {"id": "b", "text": "vec-b"},
        {"id": "c", "text": "vec-c"},
    ]

    reranker.model = None
    reranker.load_model = MagicMock()
    
    # Since rerank is now AsyncMock (from fixture), setting return_value 
    # sets the value returned when awaited.
    reranker.rerank.return_value = [
        {"id": "b"}, {"id": "a"}
    ]

    result = await hybrid.retrieve(
        query="fusion",
        project_id="p1",
        selected_sectors=["s1"],
        use_reranking=True
    )

    assert len(result["results"]) == 2
    reranker.load_model.assert_called_once()
    reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_reranker_load_failure(hybrid, mock_retrievers):
    bm25, vector, reranker = mock_retrievers

    vector.retrieve.return_value = [{"id": "x"}]
    reranker.model = None
    reranker.load_model.side_effect = Exception("fail")

    result = await hybrid.retrieve(
        query="fail",
        project_id="p1",
        selected_sectors=[],
        dense_weight=1.0,
        sparse_weight=0.0,
        use_reranking=True
    )

    # Fallback to pre-rerank results
    assert len(result["results"]) == 1


def test_rrf_fusion_logic(hybrid):
    list_a = [{"id": "1"}, {"id": "2"}]
    list_b = [{"id": "2"}, {"id": "3"}]

    fused = hybrid._rrf_fusion(list_a, list_b, k=60)

    ids = [d["id"] for d in fused]
    assert "1" in ids
    assert "2" in ids
    assert "3" in ids
    assert "rrf_score" in fused[0]