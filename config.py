"""
AutoEDA Configuration
Centralized configuration for service settings
"""

# Service Configuration
SERVICE_PORT = 8081
SERVICE_HOST = "localhost"
SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"

# Workflow Configuration
MAX_OPTIMIZATION_ITERATIONS = 3
MAX_OPTIMIZATION_ITERATIONS = 3
TARGET_ACCURACY = 0.85
LLM_MODEL = "gemini/gemini-flash-latest"  # Verified Alias for 1.5 Flash (High Limit, No 404)  

# Agent Ports (for reference if needed)
ANALYSIS_PORT = 8001
INSIGHT_PORT = 8002
PROJECT_PORT = 8003
PREPROCESSING_PORT = 8004
FEATURE_PORT = 8005
MODEL_PORT = 8006
EVALUATION_PORT = 8007
