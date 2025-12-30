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
            role="Advanced Data Preprocessing Strategist",
            goal="Determine optimal preprocessing with KNN imputation, robust scaling, and outlier handling",
            backstory="""You are an expert in advanced data preprocessing who uses:
            - KNN Imputation for better null handling (maintains relationships)
            - Robust Scaling (resistant to outliers)
            - IQR-based outlier capping (preserves data while handling extremes)
            - Smart type conversion and validation
            """,
            llm=LLM_MODEL
        )
        
        # Truncate input to avoid token limits
        input_str = str(task.input)
        if len(input_str) > 6000:
            input_str = input_str[:6000] + "...(truncated)"
            
        t = Task(
            description=f"""
Analyze the dataset and recommend ADVANCED preprocessing strategies:

Dataset Info:
{input_str}

Provide comprehensive strategies for:

1. **Null Value Handling** (Advanced):
   For each column with nulls, recommend:
   - **knn**: KNN Imputation (best for maintaining relationships) - use for < 30% nulls
   - **median**: For numerical with outliers
   - **mean**: For numerical without outliers  
   - **mode**: For categorical
   - **drop**: If > 50% nulls

2. **Outlier Detection & Handling**:
   - Method: **"iqr_capping"** (IQR method with capping, not removal)
   - Numerical columns to apply
   - Threshold: 1.5 * IQR (standard) or 3.0 * IQR (aggressive)

3. **Scaling Strategy**:
   - **robust**: RobustScaler (best for data with outliers)
   - **standard**: StandardScaler (if no outliers)
   - **minmax**: MinMaxScaler (for neural networks)
   - Specify which columns to scale

4. **Data Validation**:
   - Check for duplicate rows (recommend drop if > 1%)
   - Negative values in columns that should be positive
   - Data type corrections needed

Return detailed JSON with:
{{
  "null_strategy": {{
    "Age": {{"method": "knn", "n_neighbors": 5}},
    "Balance": {{"method": "median"}},
    "Geography": {{"method": "mode"}}
  }},
  "outlier_strategy": {{
    "method": "iqr_capping",
    "threshold": 1.5,
    "columns": ["Balance", "CreditScore", "EstimatedSalary"]
  }},
  "scaling_strategy": {{
    "method": "robust",
    "columns": ["Age", "Balance", "CreditScore", "EstimatedSalary", "Tenure"]
  }},
  "data_validation": {{
    "remove_duplicates": true,
    "fix_negatives": ["Balance", "Age"],
    "type_corrections": {{"Tenure": "int"}}
  }}
}}

Prioritize methods that preserve data and maintain relationships.
""",
            expected_output="Detailed JSON with advanced preprocessing strategies including KNN imputation, outlier capping, and robust scaling",
            agent=strategist
        )
        
        crew = Crew(agents=[strategist], tasks=[t])
        result = crew.kickoff()
        
        return A2AResponse(
            task_id=task.task_id,
            sender="preprocessing-agent",
            status="COMPLETED",
            output={"preprocessing_strategy": str(result)}
        )
    except Exception as e:
        print(f"Error: {e}")
        raise
