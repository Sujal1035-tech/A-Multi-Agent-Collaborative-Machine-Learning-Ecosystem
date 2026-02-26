from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse
import json

from config import GEMINI_MODEL
from core.llm_utils import parse_json_from_llm


def handle(task: A2ATask, log_callback=None):
    try:
        if log_callback:
            log_callback("[EVALUATION] Starting model evaluation...")

        # --- Extract problem type and set dynamic target ---
        input_data = task.input if isinstance(task.input, dict) else {"raw": str(task.input)}
        problem_type = input_data.get("problem_type", "classification")
        model_results = input_data.get("model_results", input_data.get("models", input_data))

        if problem_type == "classification":
            target_metric = "Evaluate F1-Score/AUC if dataset is imbalanced. Otherwise Accuracy ≥ 0.85"
            analysis_request = (
                "Perform a classification analysis:\n"
                "- Assess the class distribution in the results. Are the classes imbalanced?\n"
                "- If imbalanced, focus your evaluation on F1-Score, Precision, Recall, and AUC rather than raw Accuracy. Is the minority class being predicted effectively?\n"
                "- If balanced, is accuracy above 85%? If not, which classes are hardest to predict?\n"
                "- Comment on precision vs recall trade-offs based on the F1 score."
            )
        else:
            target_metric = "R² ≥ 0.70"
            analysis_request = (
                "Perform a residual analysis:\n"
                "- Is RMSE reasonable relative to the target's range?\n"
                "- Is R² above 0.70? If not, what might be causing underfitting?\n"
                "- Are there signs of heteroscedasticity based on the error metrics?"
            )

        # Truncate model results to manage token limits
        results_str = str(model_results)[:3000]

        strategist = Agent(
            role="Model Evaluation Strategist",
            goal="Evaluate ML model performance with rigorous statistical analysis",
            backstory=(
                "You are an expert in ML model evaluation who provides actionable, "
                "data-driven insights. You analyze metrics carefully and distinguish "
                "between overfitting and genuine performance."
            ),
            llm=GEMINI_MODEL
        )

        t = Task(
            description=f"""
Evaluate the model performance results below.

## Model Results
{results_str}

## Target
{target_metric}

## Required Analysis
{analysis_request}

## Overfitting Check
- Compare train scores vs test scores. A gap > 10% suggests overfitting.
- Compare CV scores vs test scores. Large discrepancies indicate instability.

## Response Format
Return a JSON object with:
{{
  "meets_target": true/false,
  "problem_type": "{problem_type}",
  "best_model": "name of the best model",
  "key_metrics": {{
    "primary_metric": 0.0,
    "cv_score": 0.0
  }},
  "overfitting_risk": "none/low/moderate/high",
  "suggestions": ["improvement1", "improvement2"],
  "analysis_summary": "2-3 sentence summary of the evaluation"
}}
""",
            expected_output="JSON with structured evaluation, overfitting assessment, and suggestions",
            agent=strategist
        )

        crew = Crew(agents=[strategist], tasks=[t])
        result = crew.kickoff()

        if log_callback:
            log_callback("[EVALUATION] Evaluation complete.")

        # Parse to validate JSON structure
        parsed = parse_json_from_llm(str(result))

        return A2AResponse(
            task_id=task.task_id,
            sender="evaluation-agent",
            status="COMPLETED",
            output={
                "evaluation": parsed if parsed else str(result),
                "problem_type": problem_type,
                "target_metric": target_metric
            }
        )
    except Exception as e:
        if log_callback:
            log_callback(f"[EVALUATION] Error: {e}")
        print(f"Error: {e}")
        raise
