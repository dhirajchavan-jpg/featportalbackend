import pytest
from datetime import datetime
from pydantic import ValidationError

from app.schemas import (
    QueryRequest,
    StandardResponse,
    SearchFilter,
    ChatHistoryEntry,
    SourceDocument,
    RetrievalStats,
    QueryResponseData
)



def test_query_request_valid_minimal():
    """
    Summary:
        Validate that a minimal valid query request is accepted with default values.

    Explanation:
        This test ensures that the QueryRequest schema correctly applies defaults
        when optional fields are omitted, allowing basic queries to be processed
        without requiring advanced configuration.
    """
    req = QueryRequest(
        query="What is GDPR?",
        project_id="proj1"
    )

    assert req.query == "What is GDPR?"
    assert req.project_id == "proj1"
    assert req.sectors is None
    assert req.comparative_mode is True
    assert req.top_k is None



def test_query_request_empty_query_rejected():
    """
    Summary:
        Empty query strings must be rejected during request validation.

    Explanation:
        This test verifies that the schema enforces non-empty user queries,
        preventing meaningless or unsafe requests from entering the system.
    """
    with pytest.raises(ValidationError):
        QueryRequest(
            query="",
            project_id="proj1"
        )


def test_query_request_empty_sectors_list_rejected():
    """
    Summary:
        An empty sectors list must be rejected during validation.

    Explanation:
        This ensures that sector-based filtering is either omitted or explicitly
        defined, avoiding ambiguous or misleading query behavior.
    """
    with pytest.raises(ValidationError) as exc:
        QueryRequest(
            query="Compare regulations",
            project_id="proj1",
            sectors=[]
        )

    assert "sectors list cannot be empty" in str(exc.value)


@pytest.mark.parametrize("top_k", [0, 21])
def test_query_request_invalid_top_k(top_k):
    """
    Summary:
        Invalid top_k values outside the allowed range must be rejected.

    Explanation:
        This test enforces constraints on the number of retrieved documents,
        preventing performance issues and misuse of retrieval parameters.
    """
    with pytest.raises(ValidationError):
        QueryRequest(
            query="Test",
            project_id="proj1",
            top_k=top_k
        )


@pytest.mark.parametrize("dense_weight", [-0.1, 1.1])
def test_query_request_invalid_dense_weight(dense_weight):
    """
    Summary:
        Dense retrieval weight must be constrained between 0 and 1.

    Explanation:
        This ensures hybrid retrieval weighting remains valid and prevents
        unstable ranking behavior caused by invalid weight configurations.
    """
    with pytest.raises(ValidationError):
        QueryRequest(
            query="Test",
            project_id="proj1",
            dense_weight=dense_weight
        )


def test_search_filter_basic():
    """
    Summary:
        Basic search filters must be converted into valid Qdrant filter queries.

    Explanation:
        This test verifies that included sources are correctly translated into
        Qdrant-compatible filter structures for vector search.
    """
    f = SearchFilter(
        sources=["Project_1", "RBI"]
    )

    q = f.to_qdrant_filter()

    assert q["must"][0]["key"] == "source"
    assert q["must"][0]["match"]["any"] == ["Project_1", "RBI"]
    assert "must_not" not in q


def test_search_filter_with_exclusions():
    """
    Summary:
        Search filters with excluded files must generate exclusion clauses.

    Explanation:
        This ensures that specific files can be explicitly excluded from search
        results, supporting fine-grained compliance and filtering requirements.
    """
    f = SearchFilter(
        sources=["Project_1"],
        excluded_files=["bad.pdf", "old.docx"]
    )

    q = f.to_qdrant_filter()

    assert q["must_not"][0]["key"] == "file_name"
    assert "bad.pdf" in q["must_not"][0]["match"]["any"]


def test_chat_history_text_message():
    """
    Summary:
        Text-based chat history entries must store user queries and LLM answers.

    Explanation:
        This test validates that conversational interactions are correctly
        persisted for auditing, traceability, and user experience continuity.
    """
    entry = ChatHistoryEntry(
        created_at=datetime.utcnow(),
        message_type="text",
        sector="SEBI",
        user_query="Hello",
        llm_answer="Hi"
    )

    assert entry.message_type == "text"
    assert entry.user_query == "Hello"
    assert entry.llm_answer == "Hi"


def test_chat_history_file_message():
    """
    Summary:
        File-based chat history entries must store file identifiers and metadata.

    Explanation:
        This ensures that document-upload interactions are properly recorded,
        enabling traceability of file-based queries in compliance workflows.
    """
    entry = ChatHistoryEntry(
        created_at=datetime.utcnow(),
        message_type="file",
        file_id="file123",
        file_name="policy.pdf",
        sector="RBI"
    )

    assert entry.file_id == "file123"
    assert entry.file_name == "policy.pdf"


def test_standard_response_generic():
    """
    Summary:
        Standard API responses must correctly wrap generic response data.

    Explanation:
        This test verifies the consistency of API responses, ensuring that data
        and error fields are populated according to the standard response schema.
    """
    response = StandardResponse[str](
        status="success",
        status_code=200,
        data="OK"
    )

    assert response.data == "OK"
    assert response.errors is None


def test_source_document_valid():
    """
    Summary:
        Source documents must include metadata and relevance scoring.

    Explanation:
        This ensures that retrieved documents carry sufficient contextual
        information and relevance scores for explainability and ranking.
    """
    doc = SourceDocument(
        page_content="Regulation text",
        metadata={"source": "RBI", "page": 1},
        relevance_score=0.92
    )

    assert doc.relevance_score == 0.92
    assert doc.metadata["source"] == "RBI"


def test_retrieval_stats_valid():
    """
    Summary:
        Retrieval statistics must be internally consistent and valid.

    Explanation:
        This test validates that retrieval metrics correctly represent the
        retrieval process, supporting monitoring and performance analysis.
    """
    stats = RetrievalStats(
        total_chunks_searched=100,
        chunks_retrieved=5,
        sources_queried=["RBI"],
        retrieval_method="hybrid",
        reranking_applied=True
    )

    assert stats.chunks_retrieved <= stats.total_chunks_searched


def test_query_response_comparative():
    """
    Summary:
        Comparative query responses must correctly represent multi-source results.

    Explanation:
        This ensures that comparative responses correctly track sources used
        and indicate comparative mode for downstream presentation and auditing.
    """
    resp = QueryResponseData(
        result="Comparison result",
        source_documents=[],
        is_comparative=True,
        sources_used={"RBI": 3, "GDPR": 2}
    )

    assert resp.is_comparative is True
    assert "RBI" in resp.sources_used
