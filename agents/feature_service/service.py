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
        engineer = Agent(
            role="Advanced Feature Engineering Expert",
            goal="Design optimal feature engineering with polynomial features, interactions, and intelligent encoding",
            backstory="""You are an expert in advanced feature engineering who creates powerful features through:
            - Polynomial features (degree 2) for numerical columns
            - Feature interactions between important variables
            - Smart categorical encoding (one-hot for low cardinality, target for high)
            - Feature selection to remove low-importance features
            - Binning for better performance
            """,
            llm=LLM_MODEL
        )
        
        # Truncate input to avoid token limits
        input_str = str(task.input)
        if len(input_str) > 6000:
            input_str = input_str[:6000] + "...(truncated)"
            
        t = Task(
            description=f"""
Analyze the dataset and recommend ADVANCED feature engineering strategies:

Dataset Info:
{input_str}

Provide comprehensive recommendations for:

1. **Polynomial Features**: 
   - Recommend numerical columns for degree-2 polynomial transformation
   - Suggest interaction pairs (e.g., Age*Balance, CreditScore*Tenure)

2. **Feature Binning**:
   - Numerical columns to bin into quantiles (improves tree models)
   - Number of bins per column (3-10)

3. **Categorical Encoding** (Smart Strategy):
   - one_hot: Low cardinality (< 5 unique values)
   - label: Ordinal categories with natural order
   - target: High cardinality (> 10 unique values) - encodes by target mean
   - ordinal: Natural ordering (e.g., education levels)

4. **Feature Selection**:
   - Features to DROP: Low correlation, high nulls, redundant (e.g., IDs, names)
   - Keep only top 80% most important features

5. **Feature Creation**:
   - Ratio features (e.g., Balance/EstimatedSalary)
   - Aggregate features (e.g., ProductsPerTenure = NumOfProducts/Tenure)
   - Flag features (e.g., IsActiveMember_and_HasCrCard)

Return detailed JSON with:
{{
  "polynomial_features": {{
    "numerical_columns": ["Age", "Balance", "CreditScore"],
    "interaction_pairs": [["Age", "Balance"], ["CreditScore", "Tenure"]]
  }},
  "binning": {{
    "Age": {{"bins": 5, "strategy": "quantile"}},
    "Balance": {{"bins": 10, "strategy": "quantile"}}
  }},
  "categorical_encoding": {{
    "one_hot": ["Geography"],
    "label": ["Gender"],
    "target": ["Surname"],
    "ordinal": {{"Education": ["HS", "College", "Masters", "PhD"]}}
  }},
  "features_to_drop": ["RowNumber", "CustomerId", "Surname"],
  "features_to_create": [
    {{"name": "BalanceToSalaryRatio", "formula": "Balance / EstimatedSalary"}},
    {{"name": "ProductsPerYear", "formula": "NumOfProducts / max(Tenure, 1)"}},
    {{"name": "IsActiveWithCard", "formula": "IsActiveMember * HasCrCard"}}
  ],
  "feature_selection": {{
    "method": "select_k_best",
    "k_percent": 80
  }}
}}

Be specific and practical. Focus on features that will improve model performance.
""",
            expected_output="Detailed JSON with advanced feature engineering strategies including polynomials, interactions, binning, encoding, and selection",
            agent=engineer
        )
        
        crew = Crew(agents=[engineer], tasks=[t])
        result = crew.kickoff()
        
        return A2AResponse(
            task_id=task.task_id,
            sender="feature-agent",
            status="COMPLETED",
            output={"feature_strategy": str(result)}
        )
    except Exception as e:
        print(f"Error: {e}")
        raise
