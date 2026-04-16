import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.services.retrieval.vector_retriever import (
    VectorRetriever,
    get_vector_retriever
)
from app.schemas import SearchFilter, QueryUnderstanding


# -------------------------
# Helpers / Mocks
# -------------------------

class MockQdrantResult:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class MockScrollPoint:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class MockQueryUnderstanding:
    def __init__(self, confidence=0.9, authority=None, document_type=None):
        self.confidence = confidence
        self.authority = authority
        self.document_type = document_type


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def search_filter():
    return SearchFilter(
        sources=["project1"],
        excluded_files=["ignore.pdf"],
        query_understanding=None
    )


@pytest.fixture
def retriever():
    with patch("app.services.retrieval.vector_retriever.get_qdrant_client") as mock_qdrant, \
         patch("app.services.retrieval.vector_retriever.get_dense_embedder") as mock_embedder:

        # Mock Qdrant Client (Sync)
        client_instance = MagicMock()
        mock_qdrant.return_value = client_instance
        
        # Ensure search and scroll are MagicMocks (Sync)
        # The provided code calls them without 'await'
        client_instance.search = MagicMock()
        client_instance.scroll = MagicMock()

        # Mock Embedder
        embedder_instance = MagicMock()
        # The code calls await aembed_query, so we need AsyncMock
        embedder_instance.aembed_query = AsyncMock()
        mock_embedder.return_value = embedder_instance

        return VectorRetriever()


# -------------------------
# retrieve()
# -------------------------

@pytest.mark.asyncio
async def test_vector_retrieve_success(retriever, search_filter):
    # Setup Embedder (Async)
    retriever.dense_embedder.aembed_query.return_value = [0.1, 0.2, 0.3]

    # Setup Qdrant Search (Sync)
    retriever.qdrant_client.search.return_value = [
        MockQdrantResult(
            id="doc1",
            score=0.95,
            payload={
                "page_content": "Test content",
                "metadata": {"source": "project1"}
            }
        )
    ]

    results = await retriever.retrieve(
        query="test query",
        search_filter=search_filter
    )

    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] == 0.95
    assert results[0]["retrieval_type"] == "vector"


@pytest.mark.asyncio
async def test_vector_retrieve_exception_returns_empty(retriever, search_filter):
    # Simulate embedder failure
    retriever.dense_embedder.aembed_query.side_effect = Exception("Embedder failure")

    results = await retriever.retrieve(
        query="test query",
        search_filter=search_filter
    )

    assert results == []


# -------------------------
# _build_qdrant_filter()
# -------------------------

def test_build_qdrant_filter_basic(retriever):
    sf = SearchFilter(
        sources=["proj1"],
        excluded_files=["file1.pdf"],
        query_understanding=None
    )

    q_filter = retriever._build_qdrant_filter(sf)

    assert q_filter is not None
    assert q_filter.must
    assert q_filter.must_not


def test_build_qdrant_filter_query_understanding(retriever):
    sf = SearchFilter(
        sources=["proj1"],
        excluded_files=[],
        query_understanding=QueryUnderstanding(
            confidence=0.9,
            authority="RBI",
            document_type="Circular"
        )
    )

    q_filter = retriever._build_qdrant_filter(sf)

    assert q_filter is not None
    keys = [cond.key for cond in q_filter.must]

    assert "metadata.sector" in keys
    assert "metadata.document_type" in keys


def test_build_qdrant_filter_low_confidence_ignored(retriever):
    sf = SearchFilter(
        sources=[],
        excluded_files=[],
        query_understanding=QueryUnderstanding(
            confidence=0.2,
            authority="RBI"
        )
    )

    q_filter = retriever._build_qdrant_filter(sf)

    # Low confidence -> query understanding ignored -> no filters
    assert q_filter is None


# -------------------------
# _format_results()
# -------------------------

def test_format_results(retriever):
    raw_results = [
        MockQdrantResult(
            id="doc1",
            score=0.9,
            payload={
                "page_content": "hello",
                "metadata": {"a": 1}
            }
        ),
        MockQdrantResult(
            id="doc2",
            score=0.8,
            payload={
                "page_content": "world",
                "metadata": {"b": 2}
            }
        )
    ]

    formatted = retriever._format_results(raw_results)

    assert formatted[0]["rank"] == 1
    assert formatted[1]["rank"] == 2
    assert formatted[0]["retrieval_type"] == "vector"


# -------------------------
# get_chunks_by_filename()
# -------------------------

@pytest.mark.asyncio
async def test_get_chunks_by_filename_success(retriever):
    # Setup Sync Scroll
    retriever.qdrant_client.scroll.return_value = (
        [
            MockScrollPoint(
                id="chunk1",
                payload={
                    "page_content": "chunk text",
                    "metadata": {"file_name": "doc.pdf"}
                }
            )
        ],
        None
    )

    results = await retriever.get_chunks_by_filename(
        project_id="proj1",
        filename="doc.pdf"
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk1"
    assert "content" in results[0]


@pytest.mark.asyncio
async def test_get_chunks_by_filename_failure(retriever):
    retriever.qdrant_client.scroll.side_effect = Exception("Qdrant down")

    results = await retriever.get_chunks_by_filename(
        project_id="proj1",
        filename="doc.pdf"
    )

    assert results == []


# -------------------------
# Singleton
# -------------------------

def test_get_vector_retriever_singleton():
    with patch("app.services.retrieval.vector_retriever.get_qdrant_client"), \
         patch("app.services.retrieval.vector_retriever.get_dense_embedder"):

        r1 = get_vector_retriever()
        r2 = get_vector_retriever()

        assert r1 is r2