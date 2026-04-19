from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse
import json
import pandas as pd
import numpy as np

from core.llm_utils import parse_json_from_llm, get_llm
from core.data_utils import load_csv_robust, normalize_missing_markers
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# =============================================================================
# FIT / TRANSFORM — Null Handling
# =============================================================================

def fit_null_handling(X, y, strategy, target_col, log_callback=None):
    """
    Fit null handling on TRAINING data only.
    Returns (X, y, state) where state stores fill values.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    state = {"fill_values": {}}
    null_strategy = strategy.get("null_strategy", strategy.get("preprocessing_strategy", {}).get("null_strategy", {}))

    for col, config in null_strategy.items():
        if col not in X.columns or col == target_col:
            continue
            
        # SAFETY GUARD: Drop columns that are >95% missing regardless of strategy
        # Imputing a column where almost all data is missing introduces severe noise and 
        # causes mathematical errors (e.g. median = NaN).
        null_pct = X[col].isna().mean()
        if null_pct > 0.95:
             state["fill_values"][col] = "DROP"
             X = X.drop(columns=[col])
             log(f"[PREPROCESS] WARNING: Null: {col} → dropped (>95% nulls: {null_pct:.1%})")
             continue

        method = config.get("method", "median") if isinstance(config, dict) else config

        if method == "mean" and pd.api.types.is_numeric_dtype(X[col]):
            fill_val = X[col].mean()
            state["fill_values"][col] = fill_val
            X[col] = X[col].fillna(fill_val)
            log(f"[PREPROCESS] Null: {col} → mean ({fill_val:.2f})")
        elif method == "median" and pd.api.types.is_numeric_dtype(X[col]):
            fill_val = X[col].median()
            if pd.isna(fill_val): fill_val = 0.0 # Fallback for 100% nulls that sneak through
            state["fill_values"][col] = fill_val
            X[col] = X[col].fillna(fill_val)
            log(f"[PREPROCESS] Null: {col} → median ({fill_val:.2f})")
        elif method == "iterative" and pd.api.types.is_numeric_dtype(X[col]):
            # Use IterativeImputer (MICE) for this column
            try:
                # We fit on all numeric columns to predict this one
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                    imputed = imputer.fit_transform(X[numeric_cols])
                    imputed_df = pd.DataFrame(imputed, columns=numeric_cols, index=X.index)
                    # We only update the specific column requested
                    X[col] = imputed_df[col]
                    state["imputers"] = state.get("imputers", {})
                    state["imputers"][col] = {"model": imputer, "cols": numeric_cols.tolist()}
                    log(f"[PREPROCESS] Null: {col} → iterative (MICE)")
                else: # Fallback to median if it's the only numeric column
                    fill_val = X[col].median()
                    if pd.isna(fill_val): fill_val = 0.0
                    state["fill_values"][col] = fill_val
                    X[col] = X[col].fillna(fill_val)
                    log(f"[PREPROCESS] Null: {col} → median (fallback from iterative)")
            except Exception as e:
                log(f"[PREPROCESS] WARNING Iterative Imputer failed for {col}: {e}. Falling back to median.")
                fill_val = X[col].median()
                if pd.isna(fill_val): fill_val = 0.0
                state["fill_values"][col] = fill_val
                X[col] = X[col].fillna(fill_val)

        elif method == "knn" and pd.api.types.is_numeric_dtype(X[col]):
            try:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    imputer = KNNImputer(n_neighbors=5)
                    imputed = imputer.fit_transform(X[numeric_cols])
                    imputed_df = pd.DataFrame(imputed, columns=numeric_cols, index=X.index)
                    X[col] = imputed_df[col]
                    state["imputers"] = state.get("imputers", {})
                    state["imputers"][col] = {"model": imputer, "cols": numeric_cols.tolist()}
                    log(f"[PREPROCESS] Null: {col} → KNN imputer")
                else:
                    fill_val = X[col].median()
                    if pd.isna(fill_val): fill_val = 0.0
                    state["fill_values"][col] = fill_val
                    X[col] = X[col].fillna(fill_val)
                    log(f"[PREPROCESS] Null: {col} → median (fallback from knn)")
            except Exception as e:
                log(f"[PREPROCESS] WARNING KNN Imputer failed for {col}: {e}. Falling back to median.")
                fill_val = X[col].median()
                if pd.isna(fill_val): fill_val = 0.0
                state["fill_values"][col] = fill_val
                X[col] = X[col].fillna(fill_val)

        elif method == "mode":
            if len(X[col].mode()) > 0:
                fill_val = X[col].mode()[0]
                state["fill_values"][col] = fill_val
                X[col] = X[col].fillna(fill_val)
                log(f"[PREPROCESS] Null: {col} → mode ({fill_val})")
        elif method == "drop":
            before = len(X)
            mask = X[col].notna()
            X = X[mask]
            y = y[mask]
            log(f"[PREPROCESS] Null: {col} → dropped {before - len(X)} rows")

    return X, y, state


def transform_null_handling(X, state, log_callback=None):
    """Apply null handling to test data using train-fitted statistics."""
    # Simple value fills
    for col, val in state.get("fill_values", {}).items():
        if col in X.columns:
            if val == "DROP":
                X = X.drop(columns=[col])
            else:
                X[col] = X[col].fillna(val)
            
    # Advanced imputer fills (MICE/KNN)
    for col, info in state.get("imputers", {}).items():
        if col in X.columns:
            try:
                model = info["model"]
                cols = info["cols"]
                # Only use if all required cols are present
                if all(c in X.columns for c in cols):
                    imputed = model.transform(X[cols])
                    imputed_df = pd.DataFrame(imputed, columns=cols, index=X.index)
                    X[col] = imputed_df[col]
            except Exception as e:
                if log_callback: log_callback(f"[PREPROCESS] Apply imputer failed on test data for {col}: {e}")
                
    return X


# =============================================================================
# FIT / TRANSFORM — Outlier Handling
# =============================================================================

def fit_outlier_handling(X, strategy, target_col, log_callback=None):
    """
    Fit outlier capping on TRAINING data only.
    Returns (X, state) where state stores IQR clip bounds.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    state = {"clip_bounds": {}}
    outlier_strategy = strategy.get("outlier_strategy", strategy.get("preprocessing_strategy", {}).get("outlier_strategy", {}))
    method = outlier_strategy.get("method", "iqr_capping")
    columns = outlier_strategy.get("columns", [])
    threshold = outlier_strategy.get("threshold", 1.5)

    if method != "iqr_capping" or not columns:
        return X, state

    for col in columns:
        if col not in X.columns or col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue

        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        state["clip_bounds"][col] = (lower, upper)

        outliers_before = ((X[col] < lower) | (X[col] > upper)).sum()
        X[col] = X[col].clip(lower=lower, upper=upper)
        log(f"[PREPROCESS] Outlier cap {col}: [{lower:.2f}, {upper:.2f}] ({outliers_before} capped)")

    return X, state


def transform_outlier_handling(X, state, log_callback=None):
    """Apply outlier capping to test data using train bounds."""
    for col, (lower, upper) in state["clip_bounds"].items():
        if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].clip(lower=lower, upper=upper)
    return X


# =============================================================================
# MAIN HANDLER — LLM Strategy Generation (no data application)
# =============================================================================

def handle(task: A2ATask, log_callback=None, api_key=None):
    def log(msg):
        if log_callback:
            log_callback(msg)

    try:
        # Get data info from input
        analysis = task.input.get("analysis_summary", task.input)
        csv_path = task.input.get("csv_path")
        target_col = analysis.get("target_column", "")
        
        llm = get_llm(api_key)

        strategist = Agent(
            role="Smart Data Preprocessing Expert",
            goal="Analyze data stats and return ONLY valid JSON with preprocessing strategies",
            backstory="""You analyze data statistics and return preprocessing recommendations as pure JSON.
            You consider null percentages, outlier counts, and skewness to make smart decisions.
            Always return ONLY a valid JSON object, no extra text.""",
            llm=llm
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
    "column_name": {{"method": "median|mean|mode|iterative|knn|drop", "reason": "brief reason why this method"}}
  }},
  "outlier_strategy": {{
    "method": "iqr_capping",
    "threshold": 1.5,
    "columns": ["col1", "col2"],
    "reason": "brief reason why this method and these columns"
  }},
  "scaling_strategy": {{
    "method": "robust|standard|minmax",
    "columns": ["col1", "col2"],
    "reason": "brief reason why this scaling method"
  }}
}}

Rules:
- null_strategy: 
  - Use `iterative` (MICE algorithm) or `knn` for numeric columns with complex relationships where >5% data is missing.
  - Use `median` for simple skewed numeric data, `mean` for normal distribution.
  - Use `mode` for categorical. 
  - Always include a brief reason.
- outlier_strategy: Include numeric columns with >2% outliers. Explain why.
- scaling_strategy: ALWAYS prefer `robust` scaling as it handles outliers better than standard. Explain why.
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
        
        # Flatten nested 'preprocessing_strategy' common LLM hallucination
        if "preprocessing_strategy" in strategy and "null_strategy" in strategy["preprocessing_strategy"]:
            strategy = strategy["preprocessing_strategy"]
        # Ensure default keys to prevent KeyError down pipeline
        strategy.setdefault("null_strategy", {})
        strategy.setdefault("outlier_strategy", {"method": "iqr_capping", "columns": [], "threshold": 1.5, "reason": "default"})
        strategy.setdefault("scaling_strategy", {"method": "robust", "columns": [], "reason": "default"})

        log(f"[PREPROCESSING] LLM Strategy: {json.dumps(strategy, indent=2)[:500]}")

        return A2AResponse(
            task_id=task.task_id,
            sender="preprocessing-agent",
            status="COMPLETED",
            output={
                "preprocessing_strategy": strategy,
                "raw_llm_output": str(result)[:500]
            }
        )
    except Exception as e:
        log(f"[PREPROCESSING] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
