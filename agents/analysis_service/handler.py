"""
Analysis Agent Handler
Shared logic for analysis functionality
"""

import pandas as pd
from a2a.schemas import A2ATask, A2AResponse


def analyze(df: pd.DataFrame):
    """Enhanced analysis with categorical detection"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    cardinality = {col: df[col].nunique() for col in categorical_cols}
    target_column = df.columns[-1]
    
    return {
        "columns": df.columns.tolist(),
        "shape": list(df.shape),
        "missing": df.isna().sum().to_dict(),
        "data_types": {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        },
        "cardinality": cardinality,
        "target_column": target_column,
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def handle_analysis(task: A2ATask) -> A2AResponse:
    """Handle analysis task"""
    df = pd.read_csv(task.input["csv_path"])
    result = analyze(df)
    
    return A2AResponse(
        task_id=task.task_id,
        sender="analysis-agent",
        status="COMPLETED",
        output={"analysis_summary": result}
    )
