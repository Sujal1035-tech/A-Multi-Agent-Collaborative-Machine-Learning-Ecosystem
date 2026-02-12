"""
Unified AutoEDA Service - Router to All Agents
This file imports handlers from agents/ folder to avoid code duplication.
Run: uvicorn unified_service:app --port 8000
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from a2a.schemas import A2ATask, A2AResponse
from dotenv import load_dotenv
import logging

# Import all agent handlers
from agents.analysis.handler import handle_analysis
from agents.project.handler import handle_project
from agents.insight.handler import handle as handle_insight_service
from agents.preprocessing.handler import handle as handle_preprocessing_service
from agents.feature.handler import handle as handle_feature_service
from agents.model.handler import handle as handle_model_service
from agents.evaluation.handler import handle as handle_evaluation_service

# Streaming support
from core.stream_utils import run_handler_streaming

# Load environment
load_dotenv()

# Set up Groq API keys — store all 3, default to KEY_1
import os
GROQ_KEYS = {
    "1": os.getenv("GROQ_API_KEY_1", ""),
    "2": os.getenv("GROQ_API_KEY_2", ""),
    "3": os.getenv("GROQ_API_KEY_3", ""),
}
os.environ["GROQ_API_KEY"] = GROQ_KEYS["1"]
print(f"[SERVICE] Loaded {sum(1 for v in GROQ_KEYS.values() if v)} Groq API keys")

# Suppress LiteLLM warnings
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# Load config
from config import SERVICE_PORT, SERVICE_HOST

app = FastAPI(title="AutoEDA Unified Service", version="3.0")


# ============================================================================
# KEY SWAP ENDPOINT — called by orchestrator before each agent group
# ============================================================================

@app.post("/swap-key/{key_id}")
def swap_key(key_id: str):
    """Swap the active GROQ_API_KEY in this process."""
    if key_id not in GROQ_KEYS or not GROQ_KEYS[key_id]:
        return {"status": "error", "message": f"Key {key_id} not found"}
    os.environ["GROQ_API_KEY"] = GROQ_KEYS[key_id]
    # Also set for litellm
    try:
        import litellm
        litellm.api_key = GROQ_KEYS[key_id]
    except Exception:
        pass
    masked = GROQ_KEYS[key_id][:8] + "..."
    print(f"[SERVICE] Swapped to GROQ_API_KEY_{key_id} ({masked})")
    return {"status": "ok", "key": key_id}

# ============================================================================
# AGENT ENDPOINTS - All import from agents/ folder (NO DUPLICATION!)
# ============================================================================

@app.post("/a2a/analysis")
def analysis_endpoint(task: A2ATask) -> A2AResponse:
    """Analysis - imports from agents/analysis/handler.py"""
    return handle_analysis(task)

@app.post("/a2a/insight")
def insight_endpoint(task: A2ATask) -> A2AResponse:
    """Insight - imports from agents/insight/handler.py"""
    return handle_insight_service(task)

@app.post("/a2a/project")
def project_endpoint(task: A2ATask) -> A2AResponse:
    """Project - imports from agents/project/handler.py"""
    return handle_project(task)

@app.post("/a2a/preprocessing")
def preprocessing_endpoint(task: A2ATask) -> A2AResponse:
    """Preprocessing - imports from agents/preprocessing/handler.py"""
    return handle_preprocessing_service(task)

@app.post("/a2a/feature")
def feature_endpoint(task: A2ATask) -> A2AResponse:
    """Feature Engineering - imports from agents/feature/handler.py"""
    return handle_feature_service(task)

@app.post("/a2a/model")
def model_endpoint(task: A2ATask) -> A2AResponse:
    """Model Training - imports from agents/model/handler.py"""
    return handle_model_service(task)

@app.post("/a2a/evaluation")
def evaluation_endpoint(task: A2ATask) -> A2AResponse:
    """Evaluation - imports from agents/evaluation/handler.py"""
    return handle_evaluation_service(task)

# ============================================================================
# STREAMING ENDPOINTS - Same agents but with real-time SSE output
# ============================================================================

AGENT_HANDLERS = {
    "analysis": handle_analysis,
    "insight": handle_insight_service,
    "project": handle_project,
    "preprocessing": handle_preprocessing_service,
    "feature": handle_feature_service,
    "model": handle_model_service,
    "evaluation": handle_evaluation_service,
}

@app.post("/a2a/{agent_name}/stream")
def stream_endpoint(agent_name: str, task: A2ATask):
    """Stream any agent's output as SSE events"""
    handler = AGENT_HANDLERS.get(agent_name)
    if not handler:
        return {"error": f"Unknown agent: {agent_name}"}
    return StreamingResponse(
        run_handler_streaming(handler, task),
        media_type="text/event-stream"
    )

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "AutoML Unified Service",
        "version": "3.0",
        "description": "Router to all agents - with streaming support!",
        "agents": 7,
        "port": SERVICE_PORT,
        "host": SERVICE_HOST,
        "endpoints": [
            "/a2a/analysis",
            "/a2a/insight",
            "/a2a/project",
            "/a2a/preprocessing",
            "/a2a/feature",
            "/a2a/model",
            "/a2a/evaluation"
        ],
        "streaming": "/a2a/{agent_name}/stream"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "unified"}
