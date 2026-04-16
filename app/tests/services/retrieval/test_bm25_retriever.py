import pytest
from unittest.mock import MagicMock, AsyncMock
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.schemas import SearchFilter

# ------------------------
# Helpers
# ------------------------

class FakeSparseEmbedder:
    def __init__(self):
        self.bm25_index = True

    # Return a dictionary as expected by the source code keys() call
    def get_sparse_embedding(self, query: str):
        return {1: 0.1, 2: 0.2, 3: 0.3}

class FakeQdrantResult:
    def __init__(self, _id, score):
        self.id = _id
        self.score = score
        self.payload = {
            "page_content": f"content {_id}",
            "metadata": {"source": "test"}
        }

# ------------------------
# Fixtures
# ------------------------

@pytest.fixture
def search_filter():
    return SearchFilter(
        sources=["proj1", "proj2"],
        excluded_files=[]
    )

# ------------------------
# Tests
# ------------------------

@pytest.mark.asyncio
async def test_returns_empty_when_no_targets(monkeypatch, search_filter):
    retriever = BM25Retriever()
    search_filter.sources = []

    results = await retriever.retrieve(
        query="test",
        search_filter=search_filter
    )
    assert results == []

@pytest.mark.asyncio
async def test_skips_missing_sparse_embedder(monkeypatch, search_filter):
    retriever = BM25Retriever()

    monkeypatch.setattr(
        "app.services.retrieval.bm25_retriever.get_sparse_embedder",
        lambda _: None
    )

    retriever.qdrant_client = MagicMock()

    results = await retriever.retrieve(
        query="test",
        search_filter=search_filter
    )
    assert results == []

@pytest.mark.asyncio
async def test_parallel_search_and_dedup(monkeypatch, search_filter):
    retriever = BM25Retriever()

    # Correctly patch get_sparse_embedder
    monkeypatch.setattr(
        "app.services.retrieval.bm25_retriever.get_sparse_embedder",
        lambda _: FakeSparseEmbedder()
    )

    # Patch qdrant_client
    retriever.qdrant_client = MagicMock()
    
    # FIX: Use MagicMock (Sync) instead of AsyncMock
    # The source code calls `self.qdrant_client.search(...)` without await.
    retriever.qdrant_client.search = MagicMock(side_effect=[
        [FakeQdrantResult("1", 0.9), FakeQdrantResult("2", 0.8)],
        [FakeQdrantResult("2", 0.8), FakeQdrantResult("3", 0.7)]
    ])

    results = await retriever.retrieve(
        query="test",
        search_filter=search_filter
    )

    # Check that deduplicated IDs are returned
    ids = sorted([r["id"] for r in results])
    assert ids == ["1", "2", "3"]

def test_format_results():
    retriever = BM25Retriever()

    fake_results = [
        FakeQdrantResult("x", 0.5),
        FakeQdrantResult("y", 0.4),
    ]

    formatted = retriever._format_results(fake_results)

    assert formatted[0]["rank"] == 1
    assert formatted[0]["id"] == "x"
    assert formatted[0]["retrieval_type"] == "bm25"