"""
Shared LLM Utilities
Common functions used by multiple agents — extracted to avoid duplication.
"""

import json
import re


def parse_json_from_llm(text: str) -> dict:
    """Extract JSON from LLM response text.
    
    Handles cases where the LLM wraps JSON in markdown code blocks
    or includes extra text around it.
    """
    try:
        json_match = re.search(r'\{[\s\S]*\}', str(text))
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {}
