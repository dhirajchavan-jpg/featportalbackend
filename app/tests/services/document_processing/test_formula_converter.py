import pytest
from unittest.mock import MagicMock, patch
from app.services.document_processing.formula_converter import FormulaConverter, get_formula_converter

# --- Fixtures ---

@pytest.fixture
def converter():
    """Returns a fresh instance of FormulaConverter."""
    return FormulaConverter()

# --- Tests ---

def test_singleton_instance():
    """Test that get_formula_converter returns a singleton."""
    instance1 = get_formula_converter()
    instance2 = get_formula_converter()
    assert instance1 is instance2
    assert isinstance(instance1, FormulaConverter)

def test_detect_formulas_display(converter):
    """Test detection of display formulas."""
    text = "Here is a formula: $$ x^2 + y^2 = z^2 $$ end."
    formulas = converter.detect_formulas(text)
    
    assert len(formulas) == 1
    assert formulas[0]["type"] == "latex_display"
    assert formulas[0]["raw"] == "$$ x^2 + y^2 = z^2 $$"

def test_detect_formulas_inline(converter):
    """Test detection of inline formulas."""
    text = "Let $x$ be a variable."
    formulas = converter.detect_formulas(text)
    
    assert len(formulas) == 1
    assert formulas[0]["type"] == "latex_inline"
    assert formulas[0]["raw"] == "$x$"

def test_detect_formulas_mixed_nested(converter):
    """Test detection preventing nested matches (inline inside display)."""
    # Inline $ inside $$ should be ignored as separate inline matches
    text = "Display: $$ a = $b $$ and Inline: $c$"
    formulas = converter.detect_formulas(text)
    
    # Logic: $$...$$ is greedy for display regex
    # It should catch "$$ a = $b $$" as ONE display formula
    # And "$c$" as ONE inline formula
    
    types = [f["type"] for f in formulas]
    assert "latex_display" in types
    assert "latex_inline" in types
    assert len(formulas) == 2

def test_clean_latex(converter):
    """Test cleaning of LaTeX delimiters."""
    assert converter._clean_latex("$$x$$") == "x"
    assert converter._clean_latex(r"\begin{equation}x\end{equation}") == "x"
    assert converter._clean_latex("$x$") == "x"

def test_simple_latex_to_text(converter):
    """Test regex-based fallback conversion."""
    latex = r"\frac{a}{b} + \sqrt{c} \le \infty"
    text = converter._simple_latex_to_text(latex)
    
    # Expected transformations based on regex map
    assert "(a)/(b)" in text
    assert "sqrt(c)" in text
    assert "<=" in text
    assert "infinity" in text

def test_convert_latex_to_text_sympy_success(converter):
    """Test conversion using SymPy parsing."""
    with patch("app.services.document_processing.formula_converter.parse_latex") as mock_parse:
        mock_expr = MagicMock()
        mock_expr.__str__.return_value = "x**2 + y"
        mock_parse.return_value = mock_expr
        
        result = converter.convert_latex_to_text("$ x^2 + y $")
        
        assert "x**2 + y" in result  # Spacing added by _format_expression
        mock_parse.assert_called_once()

def test_convert_latex_to_text_fallback(converter):
    """Test fallback when SymPy fails."""
    with patch("app.services.document_processing.formula_converter.parse_latex", side_effect=Exception("parse error")):
        # Should fall back to regex replacement
        result = converter.convert_latex_to_text(r"\alpha + \beta")
        
        assert "alpha" in result
        assert "beta" in result
        assert "+" in result

def test_process_text_with_formulas(converter):
    """Test full text processing and replacement."""
    text = "Value is $x=5$."
    
    # Mock conversion to keep test deterministic
    with patch.object(converter, 'convert_latex_to_text', return_value="x = 5"):
        result = converter.process_text_with_formulas(text)
        
        assert "Value is [x = 5] ." in result["text"]
        assert result["formula_count"] == 1
        assert result["formulas"][0]["original"] == "$x=5$"

def test_detect_inline_equations(converter):
    """Test heuristic detection of non-LaTeX math."""
    text = "If x = 5 and y > 10 then..."
    matches = converter.detect_inline_equations(text)
    
    assert "x = 5" in matches
    assert "y > 10" in matches

def test_normalize_formula_spacing(converter):
    """Test spacing normalization."""
    raw = "a+b= c"
    normalized = converter.normalize_formula_spacing(raw)
    assert normalized == "a + b = c"