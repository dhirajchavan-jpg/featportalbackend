import pytest
from unittest.mock import MagicMock, patch, mock_open
from app.services.document_processing.json_builder import DocumentProcessor, get_document_processor

# --- Fixtures ---

@pytest.fixture
def mock_components():
    """Mock all Layer 1 sub-components."""
    with patch("app.services.document_processing.json_builder.get_layout_detector") as mock_layout, \
         patch("app.services.document_processing.json_builder.get_ocr_engine") as mock_ocr, \
         patch("app.services.document_processing.json_builder.get_table_parser") as mock_table, \
         patch("app.services.document_processing.json_builder.get_formula_converter") as mock_formula:
        
        # Create mock instances
        layout_instance = MagicMock()
        ocr_instance = MagicMock()
        table_instance = MagicMock()
        formula_instance = MagicMock()
        
        # Connect mocks
        mock_layout.return_value = layout_instance
        mock_ocr.return_value = ocr_instance
        mock_table.return_value = table_instance
        mock_formula.return_value = formula_instance
        
        yield layout_instance, ocr_instance, table_instance, formula_instance

@pytest.fixture
def processor(mock_components):
    """Return an initialized DocumentProcessor with mocked dependencies."""
    return DocumentProcessor()

@pytest.fixture
def mock_settings():
    with patch("app.services.document_processing.json_builder.settings") as settings:
        settings.DETECT_FORMULAS = True
        settings.JSON_OUTPUT_DIR = "/tmp/json_output"
        yield settings

# --- Tests ---

def test_singleton_instance(mock_components):
    """Verify singleton pattern."""
    p1 = get_document_processor()
    p2 = get_document_processor()
    assert p1 is p2
    assert isinstance(p1, DocumentProcessor)

def test_process_document_flow(processor, mock_components, mock_settings):
    """Test the full processing pipeline orchestration."""
    layout, ocr, table, formula = mock_components
    
    # 1. Setup Mock Returns
    layout.detect_from_pdf.return_value = [
        {"page": 1, "type": "title", "text": "Test Doc", "bbox": [0,0,100,20]}
    ]
    
    ocr.extract_text_from_pdf.return_value = [
        {"page": 1, "text": "This is page 1 with $x=5$", "avg_confidence": 0.9}
    ]
    
    table.extract_tables_from_pdf.return_value = [
        {"page": 1, "data": [["A", "B"]]}
    ]
    
    formula.process_text_with_formulas.return_value = {
        "text": "This is page 1 with formula",
        "formulas": [{"original": "$x=5$"}],
        "formula_count": 1
    }

    # 2. Mock File System
    with patch("os.makedirs"), \
         patch("builtins.open", mock_open()) as mock_file:
        
        # 3. Execute
        result = processor.process_document(
            file_path="/data/test.pdf",
            file_id="file_123",
            project_id="proj_1",
            sector="FINANCE",
            ocr_engine_name="easyocr"
        )
        
        # 4. Verifications
        
        # Metadata check
        assert result["metadata"]["file_id"] == "file_123"
        assert result["metadata"]["ocr_engine"] == "easyocr"
        assert result["metadata"]["extraction_method"] == "ocr_pdf"
        assert result["metadata"]["total_pages"] == 1
        assert result["metadata"]["has_formulas"] is True
        
        # Component calls check
        layout.detect_from_pdf.assert_called_once_with("/data/test.pdf")
        
        # Verify OCR Strategy was passed
        ocr.extract_text_from_pdf.assert_called_once_with(
            "/data/test.pdf", engine_type="easyocr"
        )
        
        # Verify Formula Processing
        formula.process_text_with_formulas.assert_called()
        
        # Structure check
        assert result["document_structure"]["title"] == "Test Doc"
        assert len(result["pages"]) == 1
        assert len(result["tables"]) == 1


def test_process_document_docx_uses_native_extractor(processor, mock_components, mock_settings):
    layout, ocr, table, formula = mock_components

    processor.docx_extractor = MagicMock()
    processor.docx_extractor.extract_docx.return_value = {
        "pages": [{
            "layout_elements": [],
            "page_number": 1,
            "text_content": "Handbook Intro",
            "ocr_confidence": 0.0,
            "tables": [],
            "sections": [{"type": "heading", "text": "Handbook Intro", "elements": []}],
        }],
        "tables": [],
        "document_structure": {
            "title": "Handbook Intro",
            "sections": [{"page": 1, "type": "heading", "text": "Handbook Intro"}],
            "has_toc": False,
            "section_hierarchy": []
        },
        "metadata": {
            "ocr_engine": "none",
            "extraction_method": "docx_native_paginated",
            "total_pages": 1,
            "total_tables": 0,
            "page_numbering": "physical_docx",
        },
    }
    formula.process_text_with_formulas.return_value = {
        "text": "Handbook Intro",
        "formulas": [],
        "formula_count": 0
    }

    with patch("os.makedirs"), patch("builtins.open", mock_open()):
        result = processor.process_document(
            file_path="/data/test.docx",
            file_id="docx_123",
            project_id="proj_1",
            sector="HR",
            ocr_engine_name="paddleocr"
        )

    processor.docx_extractor.extract_docx.assert_called_once_with("/data/test.docx")
    layout.detect_from_pdf.assert_not_called()
    table.extract_tables_from_pdf.assert_not_called()
    ocr.extract_text_from_pdf.assert_not_called()
    assert result["metadata"]["ocr_engine"] == "none"
    assert result["metadata"]["extraction_method"] == "docx_native_paginated"
    assert result["metadata"]["total_pages"] == 1
    assert result["metadata"]["page_numbering"] == "physical_docx"
    assert result["pages"][0]["text_content"] == "Handbook Intro"

def test_process_document_docx_preserves_extracted_page_numbers(processor, mock_components, mock_settings):
    _, ocr, table, formula = mock_components
    processor.docx_extractor = MagicMock()
    processor.docx_extractor.extract_docx.return_value = {
        "pages": [
            {
                "page_number": 1,
                "layout_elements": [],
                "text_content": "Page one",
                "ocr_confidence": 0.0,
                "tables": [],
                "sections": [{"type": "heading", "text": "Page one", "elements": []}],
            },
            {
                "page_number": 2,
                "layout_elements": [],
                "text_content": "Page two",
                "ocr_confidence": 0.0,
                "tables": [],
                "sections": [{"type": "heading", "text": "Page two", "elements": []}],
            },
        ],
        "tables": [],
        "document_structure": {
            "title": "Page one",
            "sections": [
                {"page": 1, "type": "heading", "text": "Page one"},
                {"page": 2, "type": "heading", "text": "Page two"},
            ],
            "has_toc": False,
            "section_hierarchy": [],
        },
        "metadata": {
            "ocr_engine": "none",
            "extraction_method": "docx_native_paginated",
            "total_pages": 2,
            "total_tables": 0,
            "page_numbering": "physical_docx",
        },
    }
    formula.process_text_with_formulas.side_effect = [
        {"text": "Page one", "formulas": [], "formula_count": 0},
        {"text": "Page two", "formulas": [], "formula_count": 0},
    ]

    with patch("os.makedirs"), patch("builtins.open", mock_open()):
        result = processor.process_document(
            file_path="/data/test.docx",
            file_id="docx_123",
            project_id="proj_1",
            sector="HR",
            ocr_engine_name="paddleocr"
        )

    ocr.extract_text_from_pdf.assert_not_called()
    assert [p["page_number"] for p in result["pages"]] == [1, 2]
    assert result["document_structure"]["sections"][1]["page"] == 2

def test_build_pages_logic(processor, mock_components):
    """Test merging of OCR, Layout, and Tables into page objects."""
    # Data Setup
    ocr_results = [
        {"page": 1, "text": "Page 1 Text"},
        {"page": 2, "text": "Page 2 Text"}
    ]
    layout_elements = [
        {"page": 1, "type": "header"},
        {"page": 3, "type": "footer"} # Page missed by OCR but found by Layout
    ]
    tables = [
        {"page": 2, "id": "t1"}
    ]
    
    pages = processor._build_pages(layout_elements, ocr_results, tables)
    
    # Assertions
    assert len(pages) == 3 # Pages 1, 2, and 3
    
    # Check Page 1
    p1 = next(p for p in pages if p["page_number"] == 1)
    assert p1["text_content"] == "Page 1 Text"
    assert len(p1["layout_elements"]) == 1
    
    # Check Page 2
    p2 = next(p for p in pages if p["page_number"] == 2)
    assert len(p2["tables"]) == 1
    
    # Check Page 3 (The one only layout found)
    p3 = next(p for p in pages if p["page_number"] == 3)
    assert p3["text_content"] == "" # Empty text since OCR missed it
    assert len(p3["layout_elements"]) == 1

def test_extract_sections(processor):
    """Test extraction of logical sections from layout elements."""
    elements = [
        {"type": "title", "text": "Main Title"}, # Start Sec 1
        {"type": "text", "text": "Intro text"},
        {"type": "heading", "text": "Chapter 1"}, # Start Sec 2
        {"type": "paragraph", "text": "Content 1"}
    ]
    
    sections = processor._extract_sections(elements, "full text")
    
    assert len(sections) == 2
    
    # Section 1: Title
    assert sections[0]["type"] == "title"
    assert len(sections[0]["elements"]) == 2 # Title + Text
    
    # Section 2: Heading
    assert sections[1]["type"] == "heading"
    assert len(sections[1]["elements"]) == 2 # Heading + Paragraph

def test_process_document_ocr_failure(processor, mock_components, mock_settings):
    """Test pipeline robustness when OCR fails."""
    _, ocr, _, _ = mock_components
    
    # Simulate OCR Crash
    ocr.extract_text_from_pdf.side_effect = Exception("OCR Engine Died")
    
    with patch("os.makedirs"), patch("builtins.open", mock_open()):
        result = processor.process_document(
            file_path="test.pdf", 
            file_id="f1", 
            project_id="p1", 
            sector="S1"
        )
        
        # Should finish but with empty pages/text
        assert result["metadata"]["total_pages"] == 0
        assert result["pages"] == []

def test_load_processed_json(processor, mock_settings):
    """Test loading existing JSON."""
    mock_data = '{"metadata": {"file_id": "123"}}'
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_data)):
        
        data = processor.load_processed_json("123")
        assert data["metadata"]["file_id"] == "123"

def test_load_processed_json_missing(processor, mock_settings):
    """Test loading missing JSON."""
    with patch("os.path.exists", return_value=False):
        data = processor.load_processed_json("missing_id")
        assert data is None
