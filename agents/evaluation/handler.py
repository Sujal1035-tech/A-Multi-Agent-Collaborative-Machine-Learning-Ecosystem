from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse

from config import GROQ_MODEL


def handle(task: A2ATask, log_callback=None):
    try:
        if log_callback:
            log_callback("[EVALUATION] Starting model evaluation...")
            
        strategist = Agent(
            role="Model Evaluation Strategist",
            goal="Evaluate model performance and suggest improvements",
            backstory="You are an expert in ML model evaluation who provides actionable insights.",
            llm=GROQ_MODEL
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
        
        if log_callback:
            log_callback("[EVALUATION] Evaluation complete.")

        return A2AResponse(
            task_id=task.task_id,
            sender="evaluation-agent",
            status="COMPLETED",
            output={"evaluation": str(result)}
        )
    except Exception as e:
        if log_callback:
            log_callback(f"[EVALUATION] Error: {e}")
        print(f"Error: {e}")
        raise
