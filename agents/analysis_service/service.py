from fastapi import FastAPI
from a2a.schemas import A2ATask, A2AResponse
from .handler import handle_analysis

app = FastAPI()

@app.post("/a2a")
def endpoint(task: A2ATask) -> A2AResponse:
    """Analysis endpoint - uses shared handler"""
    print(f"[ANALYSIS SERVICE] Received task: {task.task_id}")
    return handle_analysis(task)
