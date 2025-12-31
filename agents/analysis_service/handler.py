"""
Analysis Agent Handler
Shared logic for analysis functionality
"""

import pandas as pd
from a2a.schemas import A2ATask, A2AResponse


def analyze(df: pd.DataFrame, target_column: str = None):
    """Enhanced analysis with detailed stats for smart preprocessing"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    cardinality = {col: df[col].nunique() for col in categorical_cols}
    
    # Use provided target column or fall back to last column
    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        target_col = df.columns[-1]
    
    # Enhanced null analysis
    null_info = {}
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        null_info[col] = {"count": int(null_count), "percent": round(null_pct, 2)}
    
    # Outlier detection using IQR for numerical columns
    outlier_info = {}
    for col in numerical_cols:
        if col == target_col:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_pct = (outlier_count / len(df)) * 100
        outlier_info[col] = {
            "count": int(outlier_count), 
            "percent": round(outlier_pct, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2)
        }
    
    # Skewness for numerical columns
    skewness = {}
    for col in numerical_cols:
        try:
            skew_val = df[col].skew()
            skewness[col] = round(skew_val, 2)
        except:
            skewness[col] = 0
    
    # Cardinality for all columns (for encoding decisions)
    all_cardinality = {col: int(df[col].nunique()) for col in df.columns}
    
    return {
        "columns": df.columns.tolist(),
        "shape": list(df.shape),
        "missing": null_info,
        "data_types": {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        },
        "cardinality": all_cardinality,
        "target_column": target_col,
        "dtypes": df.dtypes.astype(str).to_dict(),
        # New fields for smart preprocessing
        "outliers": outlier_info,
        "skewness": skewness
    }


def handle_analysis(task: A2ATask) -> A2AResponse:
    """Handle analysis task"""
    df = pd.read_csv(task.input["csv_path"])
    target_column = task.input.get("target_column")  # Get target column from input
    result = analyze(df, target_column)
    
    return A2AResponse(
        task_id=task.task_id,
        sender="analysis-agent",
        status="COMPLETED",
        output={"analysis_summary": result}
    )
