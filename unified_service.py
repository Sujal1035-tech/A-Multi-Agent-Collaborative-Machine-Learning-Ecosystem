"""
Unified AutoEDA Service - Router to All Agents
This file imports handlers from agents/ folder to avoid code duplication.
Run: uvicorn unified_service:app --port 8081
"""

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from a2a.schemas import A2ATask, A2AResponse
from dotenv import load_dotenv
import logging
import threading
import traceback
import os
from contextlib import contextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse

# Define payload size limit (40 MB)
MAX_PAYLOAD_SIZE = int(os.environ.get("MAX_PAYLOAD_SIZE", 40 * 1024 * 1024))

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get('content-length'):
            content_length = int(request.headers.get('content-length'))
            if content_length > MAX_PAYLOAD_SIZE:
                return StarletteJSONResponse(
                    content={"error": f"Payload too large. Maximum allowed is {MAX_PAYLOAD_SIZE / 1024 / 1024} MB"},
                    status_code=413
                )
        return await call_next(request)

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

# Add request size limit middleware (protect against OOM from huge payloads)
app.add_middleware(RequestSizeLimitMiddleware)

# CORS middleware for web UI support
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin to connect for local web UI testing
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


def _run_with_agent_key(agent_name: str, task: A2ATask, handler, log_callback):
    """Run agent handler with proper API key and input validation."""
    validate_task_input(task, agent_name)
    key_id = _select_key_id(agent_name, task)
    api_key = GEMINI_KEYS.get(key_id)

    import inspect
    sig = inspect.signature(handler)
    if 'api_key' in sig.parameters:
        return handler(task, log_callback, api_key=api_key)
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
# FILE UPLOAD ENDPOINT
# ============================================================================

import shutil
import time
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)



@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accepts a CSV upload, creates an output folder, and saves the file."""
    if not file.filename.endswith(('.csv', '.tsv', '.txt')):
        return JSONResponse(status_code=400, content={"error": "Only CSV, TSV, or TXT files are allowed."})
    
    # Create a unique output folder for this run
    run_id = f"autoeda_output_{int(time.time())}"
    output_folder = OUTPUTS_DIR / run_id
    output_folder.mkdir(exist_ok=True)
    (output_folder / "plots").mkdir(exist_ok=True)
    (output_folder / "models").mkdir(exist_ok=True)
    (output_folder / "reports").mkdir(exist_ok=True)
    (output_folder / "stats").mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Also copy to output folder
    shutil.copy2(str(file_path), str(output_folder / "data.csv"))
        
    return {
        "message": "File uploaded successfully", 
        "csv_path": str(file_path.absolute()),
        "output_folder": str(output_folder.absolute()),
        "run_id": run_id
    }


@app.get("/list-outputs")
async def list_outputs():
    """List all output folders available for download."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return {"folders": []}
    folders = sorted([f.name for f in outputs_dir.iterdir() if f.is_dir()], reverse=True)
    return {"folders": folders}


@app.get("/download-output/{folder_name}")
async def download_output(folder_name: str):
    """Zip and download an entire output folder."""
    import zipfile
    import io
    
    outputs_dir = Path("outputs") / folder_name
    if not outputs_dir.exists():
        return JSONResponse(status_code=404, content={"error": f"Output folder '{folder_name}' not found."})
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in outputs_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(outputs_dir)
                zf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    
    from starlette.responses import Response
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={folder_name}.zip"}
    )


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

# Static file serving — MUST be after all route definitions
from starlette.staticfiles import StaticFiles
app.mount("/static/outputs", StaticFiles(directory="outputs"), name="outputs")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {SERVICE_HOST}:{SERVICE_PORT}...")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
