"""
Unified AutoEDA Service - Router to All Agents
This file imports handlers from agents/ folder to avoid code duplication.
Run: uvicorn unified_service:app --port 8081
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from a2a.schemas import A2ATask, A2AResponse
from dotenv import load_dotenv
import logging
import threading
import traceback
from contextlib import contextmanager

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

# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

logger = logging.getLogger("autoeda")
logger.setLevel(logging.INFO)

# Console handler with structured format
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(handler)

# Suppress noisy libraries
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up Gemini API keys — store all 3, default to KEY_1
import os
GEMINI_KEYS = {
    "1": os.getenv("GEMINI_API_KEY_1", "") or os.getenv("GOOGLE_API_KEY_1", "") or os.getenv("GROQ_API_KEY_1", ""),
    "2": os.getenv("GEMINI_API_KEY_2", "") or os.getenv("GOOGLE_API_KEY_2", "") or os.getenv("GROQ_API_KEY_2", ""),
    "3": os.getenv("GEMINI_API_KEY_3", "") or os.getenv("GOOGLE_API_KEY_3", "") or os.getenv("GROQ_API_KEY_3", ""),
}
os.environ["GEMINI_API_KEY"] = GEMINI_KEYS["1"]
os.environ["GOOGLE_API_KEY"] = GEMINI_KEYS["1"]
logger.info(f"Loaded {sum(1 for v in GEMINI_KEYS.values() if v)} Gemini API keys")
KEY_LOCK = threading.RLock()

# Load config
from config import SERVICE_PORT, SERVICE_HOST

app = FastAPI(title="AutoML Unified Service", version="3.1")

# CORS middleware for web UI support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GLOBAL ERROR HANDLER
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return structured JSON errors instead of crashing with 500."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path),
            "detail": "An internal error occurred. Check server logs for details."
        }
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

ALLOWED_CSV_EXTENSIONS = {'.csv', '.tsv', '.txt'}
MAX_CSV_PATH_LENGTH = 500

def validate_task_input(task: A2ATask, agent_name: str):
    """
    Validate task input to prevent path traversal and other injection attacks.
    Raises ValueError if input is invalid.
    """
    csv_path = task.input.get("csv_path", "")
    if csv_path:
        # Length check
        if len(csv_path) > MAX_CSV_PATH_LENGTH:
            raise ValueError(f"CSV path too long ({len(csv_path)} chars, max {MAX_CSV_PATH_LENGTH})")

        # Allow URLs
        if csv_path.lower().startswith(('http://', 'https://')):
            return  # URLs are OK

        # Block path traversal
        normalized = os.path.normpath(csv_path)
        if '..' in normalized:
            raise ValueError("Path traversal detected in csv_path")

        # Check file extension
        _, ext = os.path.splitext(csv_path.lower())
        if ext and ext not in ALLOWED_CSV_EXTENSIONS:
            raise ValueError(f"Invalid file extension: {ext}")


# =============================================================================
# KEY MANAGEMENT
# =============================================================================

def _select_key_id(agent_name: str, task: A2ATask) -> str | None:
    """Pick key by agent/capability to avoid global key swapping."""
    capability = (task.capability or "").strip().lower()

    if agent_name == "insight":
        if capability == "insight_generation":
            return "3"
        return "1"
    if agent_name in {"preprocessing", "feature", "evaluation"}:
        return "2"
    if agent_name == "project":
        return "3"
    return None


@contextmanager
def _key_context(key_id: str | None):
    """Serialize LLM calls while setting key in-process for thread safety."""
    if not key_id or not GEMINI_KEYS.get(key_id):
        yield
        return

    with KEY_LOCK:
        previous_gemini_key = os.environ.get("GEMINI_API_KEY", "")
        previous_google_key = os.environ.get("GOOGLE_API_KEY", "")
        try:
            os.environ["GEMINI_API_KEY"] = GEMINI_KEYS[key_id]
            os.environ["GOOGLE_API_KEY"] = GEMINI_KEYS[key_id]
            try:
                import litellm
                litellm.api_key = GEMINI_KEYS[key_id]
            except Exception:
                pass
            yield
        finally:
            os.environ["GEMINI_API_KEY"] = previous_gemini_key
            os.environ["GOOGLE_API_KEY"] = previous_google_key
            try:
                import litellm
                litellm.api_key = previous_google_key or previous_gemini_key
            except Exception:
                pass


def _run_with_agent_key(agent_name: str, task: A2ATask, handler, log_callback):
    """Run agent handler with proper API key and input validation."""
    validate_task_input(task, agent_name)
    key_id = _select_key_id(agent_name, task)
    with _key_context(key_id):
        return handler(task, log_callback)


# ============================================================================
# KEY SWAP ENDPOINT — called by orchestrator before each agent group
# ============================================================================

@app.post("/swap-key/{key_id}")
def swap_key(key_id: str):
    """Backward-compatible no-op: key is now selected per request."""
    if key_id not in GEMINI_KEYS or not GEMINI_KEYS[key_id]:
        return {"status": "error", "message": f"Key {key_id} not found"}
    return {
        "status": "ok",
        "key": key_id,
        "mode": "request-scoped",
        "message": "Global key swapping is disabled; key is selected per request."
    }
# ============================================================================
# AGENT ENDPOINTS - All import from agents/ folder (NO DUPLICATION!)
# ============================================================================

from fastapi.concurrency import run_in_threadpool

def no_op_callback(msg: str):
    """Dummy callback for non-streaming requests."""
    logger.debug(msg)

@app.post("/a2a/analysis")
async def analysis_endpoint(task: A2ATask) -> A2AResponse:
    """Analysis - imports from agents/analysis/handler.py"""
    return await run_in_threadpool(_run_with_agent_key, "analysis", task, handle_analysis, no_op_callback)

@app.post("/a2a/insight")
async def insight_endpoint(task: A2ATask) -> A2AResponse:
    """Insight - imports from agents/insight/handler.py"""
    return await run_in_threadpool(_run_with_agent_key, "insight", task, handle_insight_service, no_op_callback)

@app.post("/a2a/project")
async def project_endpoint(task: A2ATask) -> A2AResponse:
    """Project - imports from agents/project/handler.py"""
    return await run_in_threadpool(_run_with_agent_key, "project", task, handle_project, no_op_callback)

@app.post("/a2a/preprocessing")
async def preprocessing_endpoint(task: A2ATask) -> A2AResponse:
    """Preprocessing - imports from agents/preprocessing/handler.py"""
    return await run_in_threadpool(_run_with_agent_key, "preprocessing", task, handle_preprocessing_service, no_op_callback)

@app.post("/a2a/feature")
async def feature_endpoint(task: A2ATask) -> A2AResponse:
    """Feature Engineering - imports from agents/feature/handler.py"""
    return await run_in_threadpool(_run_with_agent_key, "feature", task, handle_feature_service, no_op_callback)

@app.post("/a2a/model")
async def model_endpoint(task: A2ATask) -> A2AResponse:
    """Model Training - imports from agents/model/handler.py"""
    return await run_in_threadpool(_run_with_agent_key, "model", task, handle_model_service, no_op_callback)

@app.post("/a2a/evaluation")
async def evaluation_endpoint(task: A2ATask) -> A2AResponse:
    """Evaluation - imports from agents/evaluation/handler.py"""
    return await run_in_threadpool(_run_with_agent_key, "evaluation", task, handle_evaluation_service, no_op_callback)

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
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown agent: {agent_name}"}
        )

    # Validate input before streaming
    try:
        validate_task_input(task, agent_name)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

    return StreamingResponse(
        run_handler_streaming(
            lambda t, cb: _run_with_agent_key(agent_name, t, handler, cb),
            task
        ),
        media_type="text/event-stream"
    )

# ============================================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "AutoML Unified Service",
        "version": "3.1",
        "description": "Router to all agents — with streaming, error handling, and input validation!",
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
    return {
        "status": "healthy",
        "service": "unified",
        "keys_loaded": sum(1 for v in GEMINI_KEYS.values() if v)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {SERVICE_HOST}:{SERVICE_PORT}...")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)


