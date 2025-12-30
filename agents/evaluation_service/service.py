from fastapi import FastAPI
from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse
from dotenv import load_dotenv
import os

# Load .env from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

app = FastAPI()

from config import LLM_MODEL

@app.post("/a2a")
def handle(task: A2ATask):
    try:
        strategist = Agent(
            role="Model Evaluation Strategist",
            goal="Evaluate model performance and suggest improvements",
            backstory="You are an expert in ML model evaluation who provides actionable insights.",
            llm=LLM_MODEL
        )

        t = Task(
            description=f"""
Evaluate the model performance:

{task.input}

Target Accuracy: 85%

If best model meets target:
- Congratulate and approve
- No changes needed

If below target:
- Suggest 2-3 specific improvements
- Focus on feature engineering, hyperparameters, or data quality

Return JSON with:
{{
  "meets_target": true/false,
  "suggestions": ["improvement1", "improvement2"]
}}
""",
            expected_output="JSON with evaluation and suggestions",
            agent=strategist
        )

        crew = Crew(agents=[strategist], tasks=[t])
        result = crew.kickoff()

        return A2AResponse(
            task_id=task.task_id,
            sender="evaluation-agent",
            status="COMPLETED",
            output={"evaluation": str(result)}
        )
    except Exception as e:
        print(f"Error: {e}")
        raise
