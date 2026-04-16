import pytest
from unittest.mock import MagicMock, patch, ANY
from app.services.document_processing.layout_detector import LayoutDetector, get_layout_detector
from app.config import settings  # Import settings directly
import requests

# --- Fixtures ---

@pytest.fixture
def detector():
    """Return a fresh instance of LayoutDetector."""
    return LayoutDetector()

@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    image = MagicMock()
    image.size = (100, 100)
    # Mock save method to write bytes to the buffer passed to it
    def save_side_effect(buffer, format):
        buffer.write(b'fake_image_bytes')
    image.save.side_effect = save_side_effect
    return image

# --- Tests ---

def test_singleton_instance():
    """Test singleton getter."""
    d1 = get_layout_detector()
    d2 = get_layout_detector()
    assert d1 is d2
    assert isinstance(d1, LayoutDetector)

def test_detect_from_image_success(detector, mock_image):
    """Test successful communication with Model Server."""
    
    # Mock Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "pages": [
            {
                "page": 1,
                "elements": [
                    {"type": "title", "bbox": [0, 0, 50, 10], "text": "Title"}
                ]
            }
        ]
    }
    
    with patch("app.services.document_processing.layout_detector.requests.post", return_value=mock_response) as mock_post:
        elements = detector.detect_from_image(mock_image)
        
        # Verify Request
        # New (Fixed)
        mock_post.assert_called_once_with(f"{settings.MODEL_SERVER_URLS}/layout", files=ANY, timeout=60)
        
        # Verify Parsing
        assert len(elements) == 1
        assert elements[0]["type"] == "title"
        assert "area" in elements[0]  # Area should be calculated if missing

def test_detect_from_image_server_error(detector, mock_image):
    """Test handling of 500 error from Model Server."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    with patch("app.services.document_processing.layout_detector.requests.post", return_value=mock_response):
        elements = detector.detect_from_image(mock_image)
        assert elements == []

def test_detect_from_image_connection_error(detector, mock_image):
    """Test handling of connection failure (server down)."""
    with patch("app.services.document_processing.layout_detector.requests.post", side_effect=requests.exceptions.ConnectionError):
        elements = detector.detect_from_image(mock_image)
        assert elements == []

def test_detect_from_pdf_flow(detector, mock_image):
    """Test PDF processing flow (Convert -> Loop -> Detect)."""
    
    # Mock PDF conversion to return 2 images
    with patch("app.services.document_processing.layout_detector.convert_from_path", return_value=[mock_image, mock_image]) as mock_convert:
        
        # Mock detection for each image
        # Page 1 returns 1 title, Page 2 returns 1 paragraph
        with patch.object(detector, 'detect_from_image') as mock_detect_single:
            mock_detect_single.side_effect = [
                [{"type": "title"}],      # Call 1
                [{"type": "paragraph"}]   # Call 2
            ]
            
            results = detector.detect_from_pdf("dummy.pdf")
            
            assert len(results) == 2
            assert results[0]["page"] == 1
            assert results[0]["type"] == "title"
            assert results[1]["page"] == 2
            assert results[1]["type"] == "paragraph"
            
            mock_convert.assert_called_once_with("dummy.pdf", dpi=200)

def test_detect_from_pdf_error_handling(detector):
    """Test overall PDF processing error catch."""
    with patch("app.services.document_processing.layout_detector.convert_from_path", side_effect=Exception("PDF Corrupt")):
        results = detector.detect_from_pdf("bad.pdf")
        assert results == []

def test_calculate_area(detector):
    """Test area calculation logic."""
    bbox = [0, 0, 10, 20] # 10 width, 20 height
    area = detector._calculate_area(bbox)
    assert area == 200

def test_group_by_section(detector):
    """Test grouping elements into sections."""
    detections = [
        {"type": "title", "text": "Section A", "page": 1},
        {"type": "paragraph", "text": "Content A", "page": 1},
        {"type": "heading", "text": "Section B", "page": 1},
        {"type": "paragraph", "text": "Content B", "page": 1},
    ]
    
    sections = detector.group_by_section(detections)
    
    assert len(sections) == 2
    
    # Section A check
    assert sections[0]["title"] == "Section A"
    assert len(sections[0]["elements"]) == 2 # Title + Content
    
    # Section B check
    assert sections[1]["title"] == "Section B"
    assert len(sections[1]["elements"]) == 2 # Heading + Content

def test_extract_helpers(detector):
    """Test helper extraction methods."""
    detections = [
        {"type": "table", "id": 1},
        {"type": "paragraph", "id": 2},
        {"type": "figure", "id": 3},
        {"type": "text", "id": 4}
    ]
    
    tables = detector.extract_tables(detections)
    assert len(tables) == 1
    assert tables[0]["id"] == 1
    
    text_blocks = detector.extract_text_blocks(detections)
    assert len(text_blocks) == 2 # paragraph + text
    
    figures = detector.extract_figures(detections)
    assert len(figures) == 1
    assert figures[0]["id"] == 3