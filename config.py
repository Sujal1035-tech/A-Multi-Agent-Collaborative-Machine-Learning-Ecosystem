"""
AutoEDA Configuration
Centralized configuration for service settings and per-agent LLM models
"""

import os

# Service Configuration
SERVICE_PORT = 8081
SERVICE_HOST = "localhost"
SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"

# Workflow Configuration
MAX_OPTIMIZATION_ITERATIONS = 3
TARGET_ACCURACY = 0.85

# =============================================================================
# LLM Models - Gemini (3 keys to avoid rate limits)
# =============================================================================
GEMINI_MODEL = "gemini/gemini-2.5-flash-lite"

# Per-agent key assignment:
#   GEMINI_API_KEY_1 -> Insight agent
#   GEMINI_API_KEY_2 -> Preprocessing + Feature + Evaluation agents
#   GEMINI_API_KEY_3 -> Project agent
