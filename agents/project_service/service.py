from fastapi import FastAPI
from a2a.schemas import A2ATask, A2AResponse
from .handler import handle_project
from dotenv import load_dotenv
import os

# Load .env from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

app = FastAPI()

@app.post("/a2a")
def endpoint(task: A2ATask) -> A2AResponse:
    """Project endpoint - uses shared handler"""
    print(f"[PROJECT SERVICE] Received task: {task.task_id}")
    return handle_project(task)
