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
# LLM Models — All on Groq (3 keys to avoid rate limits)
# =============================================================================
GROQ_MODEL = "groq/llama-3.3-70b-versatile"

# Per-agent key assignment:
#   GROQ_API_KEY_1 → Insight agent
#   GROQ_API_KEY_2 → Preprocessing + Feature + Evaluation agents
#   GROQ_API_KEY_3 → Project agent

# Legacy alias
LLM_MODEL = GROQ_MODEL
