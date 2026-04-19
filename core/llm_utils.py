"""
Shared LLM Utilities
Common functions used by multiple agents — extracted to avoid duplication.
"""

import json
import re
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_MODEL

def get_llm(api_key: str = None):
    """
    Get a thread-safe LLM instance for CrewAI agents.
    Uses explicitly passed API key rather than relying on global os.environ.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    # ChatGoogleGenerativeAI expects the model name without the "gemini/" prefix used by litellm
    model_name = GEMINI_MODEL.replace("gemini/", "") if GEMINI_MODEL.startswith("gemini/") else GEMINI_MODEL
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key
    )

def parse_json_from_llm(text: str) -> dict:
    """Extract JSON from LLM response text.

    Handles cases where the LLM wraps JSON in markdown code blocks
    or includes extra text around it.

    Uses balanced-brace matching to find the correct outermost JSON
    object, avoiding the previous greedy regex that could match
    content between the first '{' and last '}' of the entire text.
    """
    text = str(text)

    # Strip markdown code fences if present
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find all candidate JSON objects using balanced brace matching
    candidates = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:j + 1])
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    # Try candidates from largest to smallest (prefer the most complete object)
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return {}
