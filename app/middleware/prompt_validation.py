# app/api/middleware/prompt_validator.py
"""
Prompt Injection Validator Middleware
Validates all user inputs before processing
"""

import re
from typing import List, Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class PromptInjectionValidator:
    """
    Detects and blocks prompt injection attempts.
    """
    
    # Dangerous patterns that indicate prompt injection
    INJECTION_PATTERNS = [
        # System prompt overrides
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"ignore\s+the\s+above",
        r"disregard\s+the\s+above",
        
        # Role switching attempts
        r"you\s+are\s+now",
        r"act\s+as\s+a",
        r"pretend\s+to\s+be",
        r"simulate\s+being",
        r"roleplay\s+as",
        
        # System message manipulation
        r"system\s*:\s*",
        r"assistant\s*:\s*",
        r"<\s*system\s*>",
        r"<\s*\/\s*system\s*>",
        
        # Instruction injection
        r"new\s+instructions?",
        r"updated\s+instructions?",
        r"revised\s+instructions?",
        r"override\s+instructions?",
        
        # Jailbreak attempts
        r"DAN\s+mode",
        r"developer\s+mode",
        r"god\s+mode",
        r"sudo\s+mode",
        
        # Prompt leaking
        r"show\s+me\s+your\s+prompt",
        r"what\s+are\s+your\s+instructions",
        r"reveal\s+your\s+system\s+prompt",
        r"print\s+your\s+instructions",
        r"system\s+instructions",
        r"show\s+me\s+your\s+(prompt|instructions)",  
        r"what\s+are\s+your\s+instructions",

        
        # Script injection
        r"<script",
        r"javascript:",
        r"onerror\s*=",
        r"onclick\s*=",
        
        # Command injection
        r"\$\{.*\}",
        r"`.*`",
        r"\|\s*sh",
        r";\s*rm\s+-rf",
    ]
    
    # Suspicious repeated characters (obfuscation attempts)
    OBFUSCATION_PATTERN = r"(.)\1{10,}"  # Same character 10+ times
    
    # Maximum safe length
    MAX_QUERY_LENGTH = 2000
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self.obfuscation = re.compile(self.OBFUSCATION_PATTERN)
        logger.info("[PromptInjectionValidator] Initialized with %d patterns", len(self.patterns))
    
    def validate(self, query: str, raise_on_detection: bool = True) -> dict:
        """
        Validate query for prompt injection attempts.
        
        Args:
            query: User query to validate
            raise_on_detection: Whether to raise HTTPException on detection
            
        Returns:
            dict with validation results
        """
        issues = []
        
        # Check 1: Length check
        if len(query) > self.MAX_QUERY_LENGTH:
            issues.append(f"Query too long ({len(query)} > {self.MAX_QUERY_LENGTH})")
        
        # Check 2: Pattern matching
        for pattern in self.patterns:
            if pattern.search(query):
                issues.append(f"Suspicious pattern detected: {pattern.pattern}")
                logger.warning(f"[PromptInjection] Detected pattern: {pattern.pattern} in query: {query[:100]}...")
        
        # Check 3: Obfuscation detection
        if self.obfuscation.search(query):
            issues.append("Obfuscation attempt detected (repeated characters)")
            logger.warning(f"[PromptInjection] Obfuscation detected in query: {query[:100]}...")
        
        # Check 4: Excessive special characters
        special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s]", query)) / len(query) if query else 0
        if special_char_ratio > 0.3:  # More than 30% special characters
            issues.append(f"Excessive special characters ({special_char_ratio:.1%})")
        
        # Check 5: Multiple newlines (payload smuggling)
        if query.count('\n') > 10:
            issues.append(f"Excessive newlines ({query.count(chr(10))})")
        
        is_safe = len(issues) == 0
        
        result = {
            "is_safe": is_safe,
            "issues": issues,
            "query_length": len(query),
            "risk_level": self._calculate_risk_level(issues)
        }
        
        if not is_safe and raise_on_detection:
            logger.error(f"[PromptInjection] BLOCKED query with issues: {issues}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Potential prompt injection detected",
                    "issues": issues,
                    "message": "Your query contains patterns that may be attempting to manipulate the system. Please rephrase your question."
                }
            )
        
        return result
    
    def _calculate_risk_level(self, issues: List[str]) -> str:
        """Calculate risk level based on issues."""
        if not issues:
            return "safe"
        elif len(issues) == 1:
            return "low"
        elif len(issues) <= 3:
            return "medium"
        else:
            return "high"
    
    def sanitize(self, query: str) -> str:
        """
        Sanitize query by removing suspicious patterns.
        Use with caution - may break legitimate queries.
        """
        sanitized = query
        
        # Remove common injection markers
        sanitized = re.sub(r"<[^>]+>", "", sanitized)  # Remove HTML-like tags
        sanitized = re.sub(r"\$\{[^}]+\}", "", sanitized)  # Remove ${...}
        sanitized = re.sub(r"`[^`]+`", "", sanitized)  # Remove backticks
        
        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized)
        sanitized = sanitized.strip()
        
        return sanitized


# Singleton instance
_validator = None

def get_prompt_validator() -> PromptInjectionValidator:
    """Get or create singleton validator."""
    global _validator
    if _validator is None:
        _validator = PromptInjectionValidator()
    return _validator