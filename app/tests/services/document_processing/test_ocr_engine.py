import pytest
from unittest.mock import MagicMock, patch, ANY
from app.services.document_processing.ocr_engine import OCREngine, get_ocr_engine, PADDLE_SERVICE_URL
import requests

# --- Fixtures ---

@pytest.fixture
def mock_easyocr_class():
    """
    Mocks the EasyOCR Reader CLASS. 
    Yields the class mock so we can check instantiation counts.
    """
    with patch("app.services.document_processing.ocr_engine.easyocr.Reader") as MockReader:
        instance = MockReader.return_value
        # Mock readtext return format: [(bbox, text, confidence)]
        instance.readtext.return_value = [
            ([0,0,0,0], "Hello", 0.9),
            ([0,0,0,0], "World", 0.8)
        ]
        yield MockReader

@pytest.fixture
def mock_fitz():
    """Mocks PyMuPDF (fitz)."""
    with patch("app.services.document_processing.ocr_engine.fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        
        # Create a Mock Page
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b'fake_png_bytes'
        mock_page.get_pixmap.return_value = mock_pix
        
        # Setup document iteration
        mock_doc.__iter__.return_value = iter([mock_page])
        # Setup indexing (doc[0])
        mock_doc.__getitem__.return_value = mock_page
        
        mock_open.return_value = mock_doc
        yield mock_open, mock_doc, mock_page

@pytest.fixture
def engine():
    """Return a fresh OCREngine instance with reset singleton."""
    # Reset singleton manually for isolation
    import app.services.document_processing.ocr_engine as mod
    mod._ocr_engine = None
    return OCREngine()

# --- Tests ---

def test_singleton_instance():
    """Verify singleton pattern."""
    e1 = get_ocr_engine()
    e2 = get_ocr_engine()
    assert e1 is e2
    assert isinstance(e1, OCREngine)

def test_initialize_ocr_lazy_loading(engine, mock_easyocr_class):
    """Test that EasyOCR loads only when requested."""
    assert engine.reader is None
    
    # Trigger loading
    reader = engine._get_reader()
    
    assert reader is not None
    assert engine.reader is not None
    
    # Verify constructor called once
    mock_easyocr_class.assert_called_once()
    
    # Call again (should skip loading)
    engine.initialize_ocr()
    
    # Verify constructor still only called once (no new calls)
    mock_easyocr_class.assert_called_once()

def test_process_image_bytes_easyocr(engine, mock_easyocr_class):
    """Test local EasyOCR processing logic."""
    # Get the instance mock from the class mock
    mock_reader_instance = mock_easyocr_class.return_value
    
    # Mock return values for this specific test
    mock_reader_instance.readtext.return_value = [
        (None, "Text1", 0.9),
        (None, "Text2", 0.9)
    ]
    
    # Inject mocked instance directly into engine
    engine.reader = mock_reader_instance
    
    text, conf = engine._process_image_bytes(b'data')
    
    assert text == "Text1 Text2"
    assert conf == 0.9

def test_process_with_paddle_service_success(engine):
    """Test successful communication with PaddleOCR service."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Service returns list of dicts
    mock_response.json.return_value = [{"text": "Paddle", "conf": 0.99}, {"text": "Result", "conf": 0.98}]
    
    with patch("app.services.document_processing.ocr_engine.requests.post", return_value=mock_response) as mock_post:
        text, conf = engine._process_with_paddle_service(b'data')
        
        assert text == "Paddle Result"
        assert conf == 0.95 # Hardcoded in code currently
        
        mock_post.assert_called_once_with(PADDLE_SERVICE_URL, files=ANY, timeout=30)

def test_process_with_paddle_service_failure(engine):
    """Test PaddleOCR service failure handling."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Error"
    
    with patch("app.services.document_processing.ocr_engine.requests.post", return_value=mock_response):
        text, conf = engine._process_with_paddle_service(b'data')
        assert text == ""
        assert conf == 0.0

def test_extract_text_from_pdf_paddle_strategy(engine, mock_fitz):
    """Test full PDF extraction flow using Paddle strategy."""
    mock_open, mock_doc, mock_page = mock_fitz
    
    # Mock the internal paddle processor
    with patch.object(engine, '_process_with_paddle_service', return_value=("Paddle Page Text", 0.95)) as mock_paddle:
        
        results = engine.extract_text_from_pdf("doc.pdf", engine_type="paddleocr")
        
        assert len(results) == 1
        assert results[0]["text"] == "Paddle Page Text"
        assert results[0]["extraction_method"] == "paddle_service"
        assert results[0]["page"] == 1
        
        mock_open.assert_called_with("doc.pdf")
        mock_page.get_pixmap.assert_called()
        mock_paddle.assert_called_once()

def test_extract_text_from_pdf_easyocr_strategy(engine, mock_fitz):
    """Test full PDF extraction flow using EasyOCR strategy."""
    mock_open, mock_doc, mock_page = mock_fitz
    
    # Mock the internal easyocr processor
    with patch.object(engine, '_process_image_bytes', return_value=("EasyOCR Text", 0.8)) as mock_local:
        
        results = engine.extract_text_from_pdf("doc.pdf", engine_type="easyocr")
        
        assert len(results) == 1
        assert results[0]["text"] == "EasyOCR Text"
        assert results[0]["extraction_method"] == "easyocr_local"
        
        mock_local.assert_called_once()

def test_extract_text_from_image(engine):
    """Test extraction from a PIL image object."""
    mock_image = MagicMock()
    # Mock save to write to buffer
    def save_side_effect(buffer, format):
        buffer.write(b'img_data')
    mock_image.save.side_effect = save_side_effect
    
    with patch.object(engine, '_process_image_bytes', return_value=("Img Text", 0.9)):
        result = engine.extract_text_from_image(mock_image)
        
        assert result["text"] == "Img Text"
        mock_image.save.assert_called()

def test_extract_text_from_region(engine, mock_fitz):
    """Test extracting text from a specific bbox on a page."""
    mock_open, mock_doc, mock_page = mock_fitz
    
    bbox = [10, 10, 100, 100]
    
    with patch.object(engine, '_process_image_bytes', return_value=("Region Text", 0.9)):
        text = engine.extract_text_from_region("doc.pdf", 1, bbox)
        
        assert text == "Region Text"
        # Verify clip rect was created
        mock_page.get_pixmap.assert_called()
        # Verify kwargs sent to get_pixmap include clip
        _, kwargs = mock_page.get_pixmap.call_args
        assert "clip" in kwargs or len(mock_page.get_pixmap.call_args[0]) > 0

@pytest.mark.asyncio
async def test_async_wrappers(engine):
    """Test that async wrappers correctly await threads."""
    with patch.object(engine, 'extract_text_from_pdf', return_value=[]) as mock_sync:
        await engine.extract_text_from_pdf_async("doc.pdf")
        mock_sync.assert_called_once()