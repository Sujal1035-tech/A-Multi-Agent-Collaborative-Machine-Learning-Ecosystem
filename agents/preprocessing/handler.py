from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse
import json
import pandas as pd
import numpy as np

from config import GROQ_MODEL
from core.llm_utils import parse_json_from_llm


def apply_null_handling(df: pd.DataFrame, strategy: dict, target_col: str) -> pd.DataFrame:
    """Apply null handling based on LLM strategy"""
    print(f"[PREPROCESSING] Applying null handling strategies...")
    
    null_strategy = strategy.get("null_strategy", {})
    
    for col, config in null_strategy.items():
        if col not in df.columns or col == target_col:
            continue
            
        if df[col].isna().sum() == 0:
            continue
            
        method = config.get("method", "median") if isinstance(config, dict) else config
        
        try:
            if method == "mean" and df[col].dtype in ['int64', 'float64']:
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                print(f"  → {col}: filled with mean ({fill_val:.2f})")
                
            elif method == "median" and df[col].dtype in ['int64', 'float64']:
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                print(f"  → {col}: filled with median ({fill_val:.2f})")
                
            elif method == "mode":
                fill_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col] = df[col].fillna(fill_val)
                print(f"  → {col}: filled with mode ({fill_val})")
                
            elif method == "knn":
                # Simple KNN approximation - use median for now
                fill_val = df[col].median() if df[col].dtype in ['int64', 'float64'] else df[col].mode()[0]
                df[col] = df[col].fillna(fill_val)
                print(f"  → {col}: filled with knn-approx ({fill_val})")
                
            elif method == "drop":
                df = df.dropna(subset=[col])
                print(f"  → {col}: dropped null rows")
        except Exception as e:
            print(f"  → {col}: error - {e}")
    
    return df


def apply_outlier_handling(df: pd.DataFrame, strategy: dict, target_col: str) -> pd.DataFrame:
    """Apply outlier handling based on LLM strategy"""
    print(f"[PREPROCESSING] Applying outlier handling...")
    
    outlier_strategy = strategy.get("outlier_strategy", {})
    method = outlier_strategy.get("method", "iqr_capping")
    threshold = outlier_strategy.get("threshold", 1.5)
    columns = outlier_strategy.get("columns", [])
    
    if method != "iqr_capping" or not columns:
        return df
    
    for col in columns:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype not in ['int64', 'float64']:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        
        outliers_before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  → {col}: capped {outliers_before} outliers [{lower:.2f}, {upper:.2f}]")
    
    return df


def apply_scaling(df: pd.DataFrame, strategy: dict, target_col: str) -> pd.DataFrame:
    """Apply scaling based on LLM strategy"""
    print(f"[PREPROCESSING] Applying scaling...")
    
    scaling_strategy = strategy.get("scaling_strategy", {})
    method = scaling_strategy.get("method", "standard")
    columns = scaling_strategy.get("columns", [])
    
    if not columns:
        return df
    
    for col in columns:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype not in ['int64', 'float64']:
            continue
            
        try:
            if method == "standard":
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                print(f"  → {col}: standard scaled")
                
            elif method == "robust":
                median_val = df[col].median()
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    df[col] = (df[col] - median_val) / IQR
                print(f"  → {col}: robust scaled")
                
            elif method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                print(f"  → {col}: minmax scaled")
        except Exception as e:
            print(f"  → {col}: scaling error - {e}")
    
    return df


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle(task: A2ATask):
    try:
        # Get data info from input
        analysis = task.input.get("analysis_summary", task.input)
        csv_path = task.input.get("csv_path")
        target_col = analysis.get("target_column", "")
        
        strategist = Agent(
            role="Smart Data Preprocessing Expert",
            goal="Analyze data stats and return ONLY valid JSON with preprocessing strategies",
            backstory="""You analyze data statistics and return preprocessing recommendations as pure JSON.
            You consider null percentages, outlier counts, and skewness to make smart decisions.
            Always return ONLY a valid JSON object, no extra text.""",
            llm=GROQ_MODEL
        )
        
        # Truncate input 
        input_str = str(analysis)
        if len(input_str) > 4000:
            input_str = input_str[:4000] + "..."
            
        t = Task(
            description=f"""
Analyze these data statistics and return preprocessing strategies as JSON:

{input_str}

Return ONLY this JSON structure (no other text):
{{
  "null_strategy": {{
    "column_name": {{"method": "mean|median|mode|knn|drop", "reason": "brief reason why this method"}}
  }},
  "outlier_strategy": {{
    "method": "iqr_capping",
    "threshold": 1.5,
    "columns": ["col1", "col2"],
    "reason": "brief reason why this method and these columns"
  }},
  "scaling_strategy": {{
    "method": "standard|robust|minmax",
    "columns": ["col1", "col2"],
    "reason": "brief reason why this scaling method"
  }}
}}

Rules:
- null_strategy: Use median for skewed data (skewness > 0.5), mean for normal distribution, mode for categorical, knn for complex patterns. Always include a brief reason.
- outlier_strategy: Include columns with >5% outliers. Explain why.
- scaling_strategy: Use robust for data with outliers, standard otherwise. Explain why.
- ONLY include columns that actually exist in the dataset
- Return ONLY valid JSON
""",
            expected_output="Pure JSON with preprocessing strategies and reasoning",
            agent=strategist
        )
        
        crew = Crew(agents=[strategist], tasks=[t])
        result = crew.kickoff()
        
        # Parse LLM strategy
        strategy = parse_json_from_llm(str(result))
        print(f"[PREPROCESSING] LLM Strategy: {json.dumps(strategy, indent=2)[:500]}")
        
        # If CSV path provided, apply preprocessing
        preprocessed_data = None
        if csv_path:
            print(f"[PREPROCESSING] Loading and preprocessing data...")
            df = pd.read_csv(csv_path)
            
            # Apply smart preprocessing
            df = apply_null_handling(df, strategy, target_col)
            df = apply_outlier_handling(df, strategy, target_col)
            # Note: Scaling applied later in model training to avoid data leakage
            
            preprocessed_data = {
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "null_remaining": int(df.isna().sum().sum())
            }
            print(f"[PREPROCESSING] Done! Shape: {df.shape}, Remaining nulls: {preprocessed_data['null_remaining']}")
        
        return A2AResponse(
            task_id=task.task_id,
            sender="preprocessing-agent",
            status="COMPLETED",
            output={
                "preprocessing_strategy": strategy,
                "preprocessed_data": preprocessed_data,
                "raw_llm_output": str(result)[:500]
            }
        )
    except Exception as e:
        print(f"[PREPROCESSING] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
