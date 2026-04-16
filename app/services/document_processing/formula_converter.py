# app/services/document_processing/formula_converter.py
"""
Layer 1: Formula Converter
Detects and converts mathematical formulas to string representation.
"""

import re
from typing import List, Dict, Any, Optional
from sympy import sympify, latex
from sympy.parsing.latex import parse_latex

import logging

logger = logging.getLogger(__name__)

class FormulaConverter:
    """Convert mathematical formulas to readable string format."""
    
    def __init__(self):
        # Store compiled regex for efficiency
        logger.info("[FormulaConverter] Initializing regex patterns...")
        self.display_patterns = [
            r'\$\$.*?\$\$',
            r'\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\}',
            r'\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}',
            r'\\begin\{math\}[\s\S]*?\\end\{math\}'
        ]
        self.inline_pattern = r'\$.*?\$'
        
        self.display_regex = re.compile('|'.join(self.display_patterns), re.DOTALL)
        self.inline_regex = re.compile(self.inline_pattern, re.DOTALL)
        
        logger.info("[FormulaConverter] Initialization complete. Regex patterns compiled.")
    
    def detect_formulas(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect mathematical formulas in text, avoiding nested matches.
        
        Args:
            text: Input text
            
        Returns:
            List of detected formulas with positions
        """
        logger.info(f"[FormulaConverter] Detecting formulas in text block of length {len(text)}")
        formulas = []
        display_spans = []

        # 1. Find all display matches first
        logger.info("[FormulaConverter] Scanning for display formulas...")
        for match in self.display_regex.finditer(text):
            span = match.span()
            formulas.append({
                "type": "latex_display",
                "raw": match.group(0),
                "start": span[0],
                "end": span[1]
            })
            display_spans.append(span)
        logger.info(f"[FormulaConverter] Found {len(display_spans)} display formulas.")

        # 2. Find all inline matches
        logger.info("[FormulaConverter] Scanning for inline formulas...")
        inline_count = 0
        skipped_nested_count = 0
        
        for match in self.inline_regex.finditer(text):
            is_nested = False
            span = match.span()
            
            # Check if this match is inside a display match
            for disp_span in display_spans:
                if disp_span[0] <= span[0] and disp_span[1] >= span[1]:
                    is_nested = True
                    break
            
            if not is_nested:
                formulas.append({
                    "type": "latex_inline",
                    "raw": match.group(0),
                    "start": span[0],
                    "end": span[1]
                })
                inline_count += 1
            else:
                skipped_nested_count += 1
        
        logger.info(f"[FormulaConverter] Found {inline_count} inline formulas. Skipped {skipped_nested_count} nested matches.")
        return formulas
    
    def convert_latex_to_text(self, latex_str: str) -> str:
        """
        Convert LaTeX formula to readable text.
        
        Args:
            latex_str: LaTeX formula string
            
        Returns:
            Human-readable text representation
        """
        logger.info(f"[FormulaConverter] Converting LaTeX string (length {len(latex_str)})")
        try:
            # Clean LaTeX delimiters
            cleaned_str = self._clean_latex(latex_str)
            
            # Parse LaTeX to SymPy expression
            expr = parse_latex(cleaned_str)
            
            # Convert to readable format
            text_repr = str(expr)
            
            # Additional formatting
            text_repr = self._format_expression(text_repr)
            
            logger.info("[FormulaConverter] SymPy conversion successful.")
            return text_repr
            
        except Exception as e:
            if "antlr4" not in str(e):
                logger.info(f"[FormulaConverter] SymPy conversion failed: {e}. Attempting fallback.")
            else:
                logger.info("[FormulaConverter] antlr4 error detected. Attempting fallback.")
            
            # Fallback: simple text conversion
            fallback_result = self._simple_latex_to_text(latex_str)
            logger.info("[FormulaConverter] Fallback conversion completed.")
            return fallback_result
    
    def _clean_latex(self, latex_str: str) -> str:
        """Remove LaTeX delimiters and environments."""
        # Remove delimiters
        latex_str = re.sub(r'\$+', '', latex_str)
        latex_str = re.sub(r'\\begin\{equation\*?\}', '', latex_str)
        latex_str = re.sub(r'\\end\{equation\*?\}', '', latex_str)
        latex_str = re.sub(r'\\begin\{align\*?\}', '', latex_str)
        latex_str = re.sub(r'\\end\{align\*?\}', '', latex_str)
        latex_str = re.sub(r'\\begin\{math\}', '', latex_str)
        latex_str = re.sub(r'\\end\{math\}', '', latex_str)
        
        return latex_str.strip()
    
    def _simple_latex_to_text(self, latex_str: str) -> str:
        """Simple rule-based LaTeX to text conversion."""
        logger.info("[FormulaConverter] Starting regex-based fallback conversion.")
        latex_str = self._clean_latex(latex_str)
        
        # Common replacements - **ORDER MATTERS**
        replacements = {
            r'\\frac\{(.*?)\}\{(.*?)\}': r'(\1)/(\2)',
            r'\\sqrt\{(.*?)\}': r'sqrt(\1)',
            r'\^\{(.*?)\}': r'**(\1)',  # Handle x^{2}
            r'_\{(.*?)\}': r'_\1',      # Handle x_{i}
            r'\^': '**',                # Handle x^2
            r'\\sum': 'sum',
            r'\\int': 'integral',
            r'\\prod': 'product',
            r'\\alpha': 'alpha',
            r'\\beta': 'beta',
            r'\\gamma': 'gamma',
            r'\\delta': 'delta',
            r'\\theta': 'theta',
            r'\\lambda': 'lambda',
            r'\\mu': 'mu',
            r'\\sigma': 'sigma',
            r'\\pi': 'pi',
            r'\\infty': 'infinity',
            r'\\le': '<=',
            r'\\ge': '>=',
            r'\\ne': '!=',
            r'\\approx': '≈',
            r'&': '',                    # Remove alignment markers
            r'\\\\': ' ',                 # Replace LaTeX newlines with a space
        }
        
        for pattern, replacement in replacements.items():
            latex_str = re.sub(pattern, replacement, latex_str)
        
        # Remove remaining backslashes
        latex_str = re.sub(r'\\', '', latex_str)
        
        # --- FIX: Clean up extra whitespace ---
        latex_str = re.sub(r'\s+', ' ', latex_str).strip()
        
        return latex_str
    
    def _format_expression(self, expr_str: str) -> str:
        """Format expression for better readability."""
        
        # --- FIX: Don't add spaces around * in ** ---
        # Add spaces around +, -, /, =, <, >
        expr_str = re.sub(r'([+\-/=<>])', r' \1 ', expr_str)
        # Add spaces around * only if it's not part of **
        expr_str = re.sub(r'(?<!\*)\*(?!\*)', r' * ', expr_str)
        
        # Remove duplicate spaces
        expr_str = re.sub(r'\s+', ' ', expr_str)
        
        return expr_str.strip()
    
    def process_text_with_formulas(self, text: str) -> Dict[str, Any]:
        """
        Process text and convert all formulas.
        
        Args:
            text: Input text with formulas
            
        Returns:
            Dictionary with processed text and formula metadata
        """
        logger.info("[FormulaConverter] Processing text block for formula replacement.")
        formulas = self.detect_formulas(text)
        
        if not formulas:
            logger.info("[FormulaConverter] No formulas found in text block.")
            return {
                "text": text,
                "formulas": [],
                "formula_count": 0
            }
        
        # Sort formulas by position (reverse order to replace from end)
        formulas.sort(key=lambda x: x['start'], reverse=True)
        
        processed_formulas = []
        processed_text = text
        count = len(formulas)
        logger.info(f"[FormulaConverter] Replacing {count} formulas in reverse order.")
        
        for i, formula in enumerate(formulas):
            # Convert formula
            converted = self.convert_latex_to_text(formula['raw'])
            
            # Replace in text
            processed_text = (
                processed_text[:formula['start']] + 
                f" [{converted}] " + 
                processed_text[formula['end']:]
            )
            
            processed_formulas.append({
                "original": formula['raw'],
                "converted": converted,
                "position": formula['start']
            })
        
        # Reverse to get original order
        processed_formulas.reverse()
        
        logger.info("[FormulaConverter] Formula replacement complete.")
        
        return {
            "text": re.sub(r'\s+', ' ', processed_text).strip(), # Clean up whitespace
            "formulas": processed_formulas,
            "formula_count": len(processed_formulas)
        }
    
    def detect_inline_equations(self, text: str) -> List[str]:
        """Detect potential inline mathematical expressions."""
        logger.info("[FormulaConverter] Scanning for non-LaTeX inline equations.")
        # Pattern for expressions like: x = 5, y > 10, a + b = c
        equation_pattern = r'\b[a-zA-Z]\s*[+\-*/=<>]\s*[a-zA-Z0-9]+\b'
        matches = re.findall(equation_pattern, text)
        if matches:
            logger.info(f"[FormulaConverter] Found {len(matches)} potential inline equations.")
        return matches
    
    def normalize_formula_spacing(self, formula: str) -> str:
        """Normalize spacing in formulas."""
        # Add spaces around operators
        formula = re.sub(r'([+\-*/=])', r' \1 ', formula)
        # Remove multiple spaces
        formula = re.sub(r'\s+', ' ', formula)
        return formula.strip()


# Singleton instance
_formula_converter = None

def get_formula_converter() -> FormulaConverter:
    """Get or create singleton FormulaConverter instance."""
    global _formula_converter
    if _formula_converter is None:
        logger.info("[FormulaConverter] Creating new singleton instance.")
        _formula_converter = FormulaConverter()
    else:
        logger.info("[FormulaConverter] Returning existing singleton instance.")
    return _formula_converter