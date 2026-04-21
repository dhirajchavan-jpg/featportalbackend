import pytest
from unittest.mock import MagicMock, patch, ANY
from langchain_core.documents import Document
from app.services.chunking.hybrid_chunker import HybridChunker

# --- Fixtures ---

@pytest.fixture
def mock_embeddings():
    """Mock the embeddings model."""
    embedder = MagicMock()
    # Mock embed_documents to return a list of vectors based on input length
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
    return embedder

@pytest.fixture
def mock_settings():
    """Mock application settings."""
    with patch("app.services.chunking.hybrid_chunker.settings") as settings:
        settings.CHUNK_SIZE = 1000
        settings.CHUNK_OVERLAP = 100
        settings.ENABLE_SECTION_CHUNKING = True
        settings.ENABLE_SEMANTIC_CHUNKING = True
        yield settings

@pytest.fixture
def chunker(mock_embeddings, mock_settings):
    """Initialize HybridChunker with mocks."""
    # We patch the SemanticChunker to avoid instantiating the real one (which might need real models)
    with patch("app.services.chunking.hybrid_chunker.SemanticChunker") as MockSemantic:
        instance = MockSemantic.return_value
        instance.create_documents.side_effect = lambda texts: [
            Document(page_content=t, metadata={}) for t in texts
        ]
        return HybridChunker(mock_embeddings)

# --- Tests ---

def test_initialization(chunker, mock_settings):
    """Test if chunkers are initialized with correct settings."""
    assert chunker.chunk_size == 1000
    assert chunker.chunk_overlap == 100
    assert chunker.semantic_chunker is not None
    assert chunker.recursive_chunker is not None

def test_clean_text(chunker):
    """Test OCR artifact cleaning logic."""
    raw_text = """
    Page | 10
    This is a test text.` 500
    Rs . 1000
    broken-
    word
    123
    
    
    Extra lines.
    """
    
    cleaned = chunker._clean_text(raw_text)
    
    # Assertions based on _clean_text logic
    assert "Page | 10" not in cleaned
    assert "Rs. 500" in cleaned
    assert "Rs. 1000" in cleaned # Rs . fixed
    assert "brokenword" in cleaned # Hyphen merged
    assert "123" not in cleaned # Floating integer removed (if on own line)
    assert "\n\n\n" not in cleaned # Max 2 newlines


def test_clean_text_fixes_common_mojibake(chunker):
    raw_text = "Employeesâ€™ rights include 3â€“5 days of leave and â€œquoted textâ€\x9d."

    cleaned = chunker._clean_text(raw_text)

    assert "Employees' rights" in cleaned
    assert "3-5" in cleaned
    assert '"quoted text"' in cleaned

def test_table_to_markdown(chunker):
    """Test JSON table to Markdown conversion."""
    table_data = {
        "headers": ["Col1", "Col2"],
        "data": [
            ["Row1-1", "Row1-2"],
            ["Row2-1", "Row2-2"]
        ]
    }
    
    md = chunker._table_to_markdown(table_data)
    
    assert "| Col1 | Col2 |" in md
    assert "| --- | --- |" in md
    assert "| Row1-1 | Row1-2 |" in md

def test_chunk_tables(chunker):
    """Test processing of table pages."""
    tables = [{
        "headers": ["H1"],
        "data": [["D1"]]
    }]
    base_meta = {"file_name": "test.pdf"}
    context = "File: test.pdf"
    
    chunks = chunker._chunk_tables(tables, 1, base_meta, context)
    
    assert len(chunks) == 1
    assert "Table Data" in chunks[0].page_content
    assert "| H1 |" in chunks[0].page_content
    assert chunks[0].metadata['content_type'] == "table"


def test_chunk_tables_skips_empty_header_only_tables(chunker):
    tables = [{
        "headers": ["Outcome", "Action"],
        "data": []
    }]
    chunks = chunker._chunk_tables(tables, 1, {"file_name": "test.pdf"}, "File: test.pdf")

    assert chunks == []

def test_chunk_by_legal_clauses_basic(chunker):
    """Test regex splitting on standard legal headers."""
    text = """
    CHAPTER I
    INTRODUCTION
    1. Short title.
    This Act may be called the Test Act.
    2. Definitions.
    In this Act...
    """
    
    section_tracker = {"current_header": "Start", "last_clause_index": 0}
    chunks = chunker._chunk_by_legal_clauses(text, 1, {}, "Context", section_tracker)
    
    # Expect chunks for Chapter, Clause 1, Clause 2
    content_combined = " ".join([c.page_content for c in chunks])
    assert "CHAPTER I" in content_combined
    assert "1." in content_combined
    assert "2." in content_combined
    
    # Check Context Update
    assert "CHAPTER I" in section_tracker['current_header']

def test_chunk_by_legal_clauses_sticky_context(chunker):
    """
    Test the 'Context Injection' feature where definition headers 
    stick to subsequent list items.
    """
    text = """
    In this Act, unless the context otherwise requires:-
    (a) "Admin" means the administrator;
    (b) "User" means the person.
    """
    
    section_tracker = {"current_header": "Defs", "last_clause_index": 0}
    chunks = chunker._chunk_by_legal_clauses(text, 1, {}, "Context", section_tracker)
    
    # We expect the preamble "In this Act...-:" to be injected into (a) and (b)
    
    # Chunk (a)
    chunk_a = next(c for c in chunks if "(a)" in c.page_content)
    assert "In this Act, unless the context otherwise requires:-" in chunk_a.page_content
    assert "-> (a)" in chunk_a.page_content
    
    # Chunk (b)
    chunk_b = next(c for c in chunks if "(b)" in c.page_content)
    assert "In this Act, unless the context otherwise requires:-" in chunk_b.page_content
    assert "-> (b)" in chunk_b.page_content

def test_chunk_semantically_fallback(chunker, mock_settings):
    """Test fallback to semantic chunking when legal chunking is disabled or fails."""
    # Disable legal chunking
    mock_settings.ENABLE_SECTION_CHUNKING = False
    
    # FIX: Text must be >= 50 chars to pass the filter in `chunk_document`
    text = "This is a blob of unstructured text without numbers or headers. " * 2
    
    # Mock Semantic Chunker response
    chunker.semantic_chunker.create_documents.return_value = [
        Document(page_content=text, metadata={})
    ]
    
    chunks = chunker.chunk_document({
        "pages": [{"page_number": 1, "text_content": text}],
        "metadata": {"file_name": "test.txt"}
    })
    
    assert len(chunks) == 1
    assert chunks[0].metadata['chunk_method'] == 'semantic'
    chunker.semantic_chunker.create_documents.assert_called_once()

def test_chunk_recursive_ultimate_fallback(chunker, mock_settings):
    """Test fallback to recursive chunking when semantic also fails/disabled."""
    mock_settings.ENABLE_SECTION_CHUNKING = False
    mock_settings.ENABLE_SEMANTIC_CHUNKING = False
    
    # FIX: Text must be >= 50 chars to pass the filter in `chunk_document`
    text = "Simple text is not enough. We need more content to ensure processing happens. " * 3
    
    chunks = chunker.chunk_document({
        "pages": [{"page_number": 1, "text_content": text}],
        "metadata": {"file_name": "test.txt"}
    })
    
    assert len(chunks) == 1
    assert chunks[0].metadata['chunk_method'] == 'recursive'

def test_chunk_document_full_flow(chunker):
    """Test processing a full document with multiple pages and types."""
    # Ensure text is long enough for processing
    text_content = "CHAPTER I\nPRELIMINARY\n1. Scope.\nThis applies to all. " * 5
    
    document_json = {
        "metadata": {"file_name": "policy.pdf"},
        "pages": [
            {
                "page_number": 1,
                "text_content": text_content,
                "tables": []
            },
            {
                "page_number": 2,
                "text_content": "",
                "tables": [{"headers": ["A"], "data": [["B"]]}]
            }
        ]
    }
    
    chunks = chunker.chunk_document(document_json)
    
    # Validate Text Chunks (Page 1)
    text_chunks = [c for c in chunks if c.metadata['page'] == 1]
    assert len(text_chunks) > 0
    assert "CHAPTER I" in text_chunks[0].page_content
    
    # Validate Table Chunks (Page 2)
    table_chunks = [c for c in chunks if c.metadata['page'] == 2]
    assert len(table_chunks) == 1
    assert "Table Data" in table_chunks[0].page_content
    
    # Validate Embeddings were generated
    # The mock embedder adds 'dense_embedding' to metadata
    assert 'dense_embedding' in chunks[0].metadata


def test_docx_native_chunks_preserve_real_page_metadata(chunker):
    text_content = "POLICY OVERVIEW\nThis handbook explains the company policies in detail. " * 3

    chunks = chunker.chunk_document({
        "metadata": {"file_name": "policy.docx", "page_numbering": "physical_docx"},
        "pages": [{
            "page_number": 3,
            "text_content": text_content,
            "tables": []
        }]
    })

    assert len(chunks) > 0
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)


def test_uppercase_headings_improve_section_context(chunker):
    text = """
    EMPLOYEE ENGAGEMENT
    RECOGNITION & REWARDS
    Objective
    To build a culture of appreciation.
    1. Eligibility.
    All employees are eligible.
    """

    section_tracker = {"current_header": "Preamble / Introduction", "last_clause_index": 0}
    chunks = chunker._chunk_by_legal_clauses(text, 2, {}, "File: handbook.pdf", section_tracker)

    assert len(chunks) > 0
    assert "EMPLOYEE ENGAGEMENT / RECOGNITION & REWARDS" in chunks[0].page_content

def test_generate_embeddings_batched(chunker, mock_embeddings):
    """Test batch embedding generation."""
    docs = [Document(page_content=f"Text {i}") for i in range(150)]
    
    # Batch size 50 -> 3 calls
    chunker._generate_embeddings_batched(docs, batch_size=50)
    
    assert mock_embeddings.embed_documents.call_count == 3
    assert 'dense_embedding' in docs[0].metadata

def test_chunk_statistics(chunker):
    """Test stats calculation."""
    docs = [
        Document(page_content="12345"), # len 5
        Document(page_content="1234567890") # len 10
    ]
    
    stats = chunker.get_chunk_statistics(docs)
    
    assert stats["total_chunks"] == 2
    assert stats["avg_length"] == 7.5
    assert stats["min_length"] == 5
    assert stats["max_length"] == 10
