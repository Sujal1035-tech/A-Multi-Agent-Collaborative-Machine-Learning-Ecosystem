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
            role="Senior Data Analyst",
            goal="Generate precise, actionable insights from dataset analysis in a well-structured format",
            backstory="""You are a senior data analyst with expertise in statistical analysis and machine learning. 
            You excel at presenting complex findings in clear, structured reports that are easy to understand.
            Always organize your insights into logical sections with proper formatting.""",
            llm=LLM_MODEL
        )
        
        # Truncate input to avoid token limits
        input_str = str(task.input)
        if len(input_str) > 6000:
            input_str = input_str[:6000] + "...(truncated)"
            
        t = Task(
            description=f"""Analyze the following data and generate structured insights:

{input_str}

IMPORTANT: Structure your response using the following format:

## 📊 Executive Summary
- Provide 2-3 bullet points summarizing the most important findings

## 📋 Data Quality Assessment
| Metric | Value | Status |
|--------|-------|--------|
(Include: completeness, missing values, outliers, data types)

## 🔍 Feature Analysis
For each key feature, provide:
- **Feature Name**: Brief description
  - Distribution: (normal/skewed/bimodal)
  - Key Statistics: mean, median, std
  - Notable patterns

## 📈 Model Performance Summary
| Model | Training Score | CV Score | Recommendation |
|-------|----------------|----------|----------------|
(Rank models from best to worst based on CV score)

## ✅ Key Recommendations
1. [Numbered actionable recommendations]
2. [Based on the analysis findings]
3. [Prioritized by importance]

## ⚠️ Potential Issues & Warnings
- List any data quality concerns
- Highlight areas needing attention

Use clear formatting, bullet points, and tables where appropriate.""",
            expected_output="""A well-structured analysis report with:
1. Executive Summary (2-3 key takeaways)
2. Data Quality table with metrics
3. Feature analysis with statistics
4. Model performance comparison table
5. Numbered recommendations
6. Warnings section if applicable""",
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
