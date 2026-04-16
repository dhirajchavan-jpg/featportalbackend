import pytest
from unittest.mock import MagicMock, patch, mock_open
import pickle
import os
from app.services.embedding.sparse_embedder import SparseEmbedder

# --- Fixtures ---

@pytest.fixture
def embedder():
    """Return a fresh SparseEmbedder instance."""
    return SparseEmbedder()

# --- Tests ---

def test_initialization(embedder):
    """Test initialization state."""
    assert embedder.bm25_index is None
    assert embedder.corpus == []
    assert embedder.vocab == {}

def test_tokenize(embedder):
    """Test tokenization logic (lowercase, alphanumeric, len > 2)."""
    text = "Hello World! This is a test."
    tokens = embedder._tokenize(text)
    
    # "is" (len 2) and "a" (len 1) should be removed
    expected = ["hello", "world", "this", "test"]
    assert tokens == expected

def test_build_index(embedder):
    """Test building BM25 index and vocabulary."""
    corpus = [
        "apple banana orange",
        "apple banana",
        "orange"
    ]
    
    embedder.build_index(corpus)
    
    assert len(embedder.corpus) == 3
    assert embedder.bm25_index is not None
    assert "apple" in embedder.vocab
    assert "banana" in embedder.vocab
    assert embedder.vocab_size == 3

def test_get_sparse_embedding_logic(embedder):
    """Test sparse vector generation with IDF scores."""
    # Setup a slightly larger corpus to ensure stable IDF math
    # "rare" appears 1 time (Doc 1)
    # "common" appears 3 times (Docs 2, 3, 4)
    corpus = [
        "rare word", 
        "common word",
        "common thing",
        "common place"
    ]
    embedder.build_index(corpus)
    
    # Query: "rare common"
    vector = embedder.get_sparse_embedding("rare common")
    
    # Get IDs
    id_rare = embedder.vocab["rare"]
    id_common = embedder.vocab["common"]
    
    # Check that vector contains both
    assert id_rare in vector
    assert id_common in vector
    
    # Logic check: Scores should be positive
    # With 4 docs, "rare" (1/4) implies high info, "common" (3/4) implies low info
    assert vector[id_rare] > 0.0
    assert vector[id_common] > 0.0
    
    # "rare" should technically have a higher score than "common"
    assert vector[id_rare] > vector[id_common]

def test_get_sparse_embedding_no_index(embedder):
    """Test safe return when index isn't built."""
    vec = embedder.get_sparse_embedding("test")
    assert vec == {}

def test_get_sparse_embedding_unknown_word(embedder):
    """Test handling of words not in vocabulary."""
    embedder.build_index(["known word"])
    vec = embedder.get_sparse_embedding("unknown")
    assert vec == {}

def test_save_index(embedder):
    """Test saving index to disk (pickle)."""
    embedder.build_index(["test"])
    
    with patch("builtins.open", mock_open()) as mock_file, \
         patch("os.makedirs"):
        
        embedder.save_index("/tmp/index.pkl")
        
        # Verify pickle dump called
        mock_file.assert_called_with("/tmp/index.pkl", "wb")

def test_load_index_success(embedder):
    """Test loading index from disk."""
    fake_data = {
        'bm25_index': "fake_bm25",
        'corpus': ["c"],
        'tokenized_corpus': [["c"]],
        'vocab': {"c": 0},
        'vocab_size': 1
    }
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=pickle.dumps(fake_data))):
        
        embedder.load_index("/tmp/index.pkl")
        
        assert embedder.bm25_index == "fake_bm25"
        assert embedder.vocab_size == 1

def test_load_index_not_found(embedder):
    """Test loading non-existent file."""
    with patch("os.path.exists", return_value=False):
        embedder.load_index("missing.pkl")
        assert embedder.bm25_index is None

def test_update_index(embedder):
    """Test updating existing index with new documents."""
    # 1. Initial Build
    embedder.build_index(["apple"])
    assert "banana" not in embedder.vocab
    
    # 2. Update
    embedder.update_index(["banana"])
    
    # 3. Verify
    assert len(embedder.corpus) == 2
    assert "banana" in embedder.vocab
    assert embedder.vocab_size == 2
    # Verify BM25 was rebuilt (it should be a valid object)
    assert embedder.bm25_index is not None

def test_update_index_empty(embedder):
    """Test update with empty list (no-op)."""
    embedder.build_index(["test"])
    initial_size = len(embedder.corpus)
    
    embedder.update_index([])
    assert len(embedder.corpus) == initial_size