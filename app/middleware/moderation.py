# app/core/moderation.py

from app.core.llm_provider import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import logging
logger = logging.getLogger(__name__)    

def is_compliance_related(text_to_check: str) -> bool:
    """
    Uses the main LLM to determine if a piece of text is both related to compliance
    and is appropriate.
    """
    logger.debug(f"Moderating text: '{text_to_check[:100]}...'")
    
    llm = get_llm()

    # --- ENHANCED PROMPT TEMPLATE ---
    # This prompt now has two rules: relevance AND safety.
    moderation_prompt_template = """
    You are a strict content moderator for a professional compliance application.
    Your task is to determine if the following text is BOTH relevant and appropriate.
    The text is considered VALID only if it is related to legal, financial, medical, or corporate compliance, regulations, audits, or governance AND is not harmful, unethical, inappropriate, or malicious.
    Do not answer the question in the text. Do not be conversational.
    Answer with only the single word 'Yes' if the text is BOTH relevant and appropriate.
    Answer with the single word 'No' if the text is either irrelevant OR inappropriate.

    Text to analyze: "{text}"

    Your single-word answer:
    """
    
    moderation_prompt = PromptTemplate.from_template(moderation_prompt_template)
    
    chain = moderation_prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"text": text_to_check})
        
        cleaned_response = response.strip().capitalize()
        logger.debug(f"Moderation response from LLM: '{cleaned_response}'")
        
        return cleaned_response == "Yes"
        
    except Exception as e:
        logger.error(f"ERROR during moderation LLM call: {e}")
        # Fail securely: if the check fails, assume the content is invalid.
        return False