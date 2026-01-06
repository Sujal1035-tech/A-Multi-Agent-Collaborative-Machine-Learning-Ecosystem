"""
Unified AutoEDA Service - Router to All Agents
This file imports handlers from agents/ folder to avoid code duplication.
Run: uvicorn unified_service:app --port 8000
"""

from fastapi import FastAPI
from a2a.schemas import A2ATask, A2AResponse
from dotenv import load_dotenv
import logging

# Import agent handlers (no code duplication!)
from agents.analysis_service.handler import handle_analysis
from agents.project_service.handler import handle_project

# Import service handlers directly
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Load from individual services
from agents.insight_service.service import handle as handle_insight_service
from agents.preprocessing_service.service import handle as handle_preprocessing_service
from agents.feature_service.service import handle as handle_feature_service
from agents.model_service.service import handle as handle_model_service
from agents.evaluation_service.service import handle as handle_evaluation_service

# Load environment
load_dotenv()

# Suppress LiteLLM warnings
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# Load config
from config import SERVICE_PORT, SERVICE_HOST

app = FastAPI(title="AutoEDA Unified Service", version="3.0")

# ============================================================================
# AGENT ENDPOINTS - All import from agents/ folder (NO DUPLICATION!)
# ============================================================================

@app.post("/a2a/analysis")
def analysis_endpoint(task: A2ATask) -> A2AResponse:
    """Analysis - imports from agents/analysis_service/handler.py"""
    return handle_analysis(task)

@app.post("/a2a/insight")
def insight_endpoint(task: A2ATask) -> A2AResponse:
    """Insight - imports from agents/insight_service/service.py"""
    return handle_insight_service(task)

@app.post("/a2a/project")
def project_endpoint(task: A2ATask) -> A2AResponse:
    """Project - imports from agents/project_service/handler.py"""
    return handle_project(task)

@app.post("/a2a/preprocessing")
def preprocessing_endpoint(task: A2ATask) -> A2AResponse:
    """Preprocessing - imports from agents/preprocessing_service/service.py"""
    return handle_preprocessing_service(task)

@app.post("/a2a/feature")
def feature_endpoint(task: A2ATask) -> A2AResponse:
    """Feature Engineering - imports from agents/feature_service/service.py"""
    return handle_feature_service(task)

@app.post("/a2a/model")
def model_endpoint(task: A2ATask) -> A2AResponse:
    """Model Training - imports from agents/model_service/service.py"""
    return handle_model_service(task)

@app.post("/a2a/evaluation")
def evaluation_endpoint(task: A2ATask) -> A2AResponse:
    """Evaluation - imports from agents/evaluation_service/service.py"""
    return handle_evaluation_service(task)

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "AutoML Unified Service",
        "version": "3.0",
        "description": "Router to all agents - NO code duplication!",
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
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "unified"}
