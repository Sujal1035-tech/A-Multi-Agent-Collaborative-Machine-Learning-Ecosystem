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
        analyst = Agent(
            role="Data Analyst",
            goal="Generate insights from dataset analysis",
            backstory="You are an expert data analyst who finds meaningful patterns in data.",
            llm=LLM_MODEL
        )
        
        # Truncate input to avoid token limits
        input_str = str(task.input)
        if len(input_str) > 6000:
            input_str = input_str[:6000] + "...(truncated)"
            
        t = Task(
            description=f"Generate insights from:\n{input_str}",
            expected_output="A list of key insights and patterns found in the data analysis",
            agent=analyst
        )
        
        crew = Crew(agents=[analyst], tasks=[t])
        insights = crew.kickoff()
        
        return A2AResponse(
            task_id=task.task_id,
            sender="insight-agent",
            status="COMPLETED",
            output={"insights": str(insights)}
        )
    except Exception as e:
        print(f"Error: {e}")
        raise
