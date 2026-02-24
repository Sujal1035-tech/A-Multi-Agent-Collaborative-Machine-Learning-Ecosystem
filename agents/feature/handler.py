from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse
import json
import pandas as pd
import numpy as np

from config import GROQ_MODEL
from core.llm_utils import parse_json_from_llm


def apply_encoding(df: pd.DataFrame, strategy: dict, target_col: str, log_callback=None) -> pd.DataFrame:
    """Apply smart encoding based on LLM strategy"""
    def log(msg):
        if log_callback:
            log_callback(msg)

    log(f"[FEATURE] Applying smart encoding strategies...")
    
    encoding_strategy = strategy.get("encoding_strategy", {})
    
    # One-hot encoding (low cardinality)
    onehot_cols = encoding_strategy.get("onehot", [])
    for col in onehot_cols:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype == 'object':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            log(f"  → {col}: one-hot encoded ({len(dummies.columns)} new columns)")
    
    # Label encoding (ordinal or binary)
    label_cols = encoding_strategy.get("label", [])
    for col in label_cols:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            mapping = {v: i for i, v in enumerate(unique_vals)}
            df[col] = df[col].map(mapping)
            log(f"  → {col}: label encoded ({len(mapping)} classes)")
    
    # Target encoding (high cardinality) - simplified version
    target_cols = encoding_strategy.get("target", [])
    for col in target_cols:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype == 'object':
            # For target encoding, use frequency as proxy (safer)
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
            log(f"  → {col}: frequency encoded (proxy for target)")
    
    # Handle any remaining object columns with label encoding
    for col in df.select_dtypes(include=['object']).columns:
        if col == target_col:
            continue
        unique_vals = df[col].unique()
        mapping = {v: i for i, v in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)
        log(f"  → {col}: auto label encoded ({len(mapping)} classes)")
    
    return df


def drop_features(df: pd.DataFrame, strategy: dict, target_col: str, log_callback=None) -> pd.DataFrame:
    """Drop features recommended by LLM"""
    def log(msg):
        if log_callback:
            log_callback(msg)

    features_to_drop = strategy.get("features_to_drop", [])
    if not isinstance(features_to_drop, list):
        features_to_drop = []
    
    for col in features_to_drop:
        if not isinstance(col, str):
            continue
        if col in df.columns and col != target_col:
            df = df.drop(col, axis=1)
            log(f"  → Dropped: {col}")
    
    return df


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle(task: A2ATask, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)

    try:
        # Get analysis info
        analysis = task.input.get("analysis_summary", task.input)
        csv_path = task.input.get("csv_path")
        target_col = analysis.get("target_column", "")
        cardinality = analysis.get("cardinality", {})

        # Load CSV to get actual column dtypes (used later to filter LLM mistakes)
        df_dtypes = {}
        if csv_path:
            try:
                _df = pd.read_csv(csv_path, nrows=5)
                df_dtypes = {col: str(_df[col].dtype) for col in _df.columns}
            except Exception:
                pass
        
        engineer = Agent(
            role="Smart Feature Engineering Expert",
            goal="Analyze data and return ONLY valid JSON with encoding strategies",
            backstory="""You analyze data cardinality and types to recommend optimal encoding.
            - One-hot for low cardinality (<5 unique values)
            - Label for binary or ordinal data
            - Target/frequency encoding for high cardinality (>10 unique)
            Always return ONLY a valid JSON object.""",
            llm=GROQ_MODEL
        )
        
        # Truncate input
        input_str = str(analysis)
        if len(input_str) > 4000:
            input_str = input_str[:4000] + "..."
            
        t = Task(
            description=f"""
Analyze these data stats and return encoding strategies as JSON:

{input_str}

Return ONLY this JSON structure:
{{
  "encoding_strategy": {{
    "onehot": ["low_cardinality_categorical_cols"],
    "label": ["binary_or_ordinal_categorical_cols"],
    "target": ["high_cardinality_categorical_cols"],
    "reason": "brief explanation of why each column was assigned that encoding type"
  }},
  "features_to_drop": ["id_cols", "name_cols", "useless_cols"],
  "drop_reason": "brief explanation of why these features should be dropped"
}}

Rules:
- CRITICAL: ONLY encode columns that are CATEGORICAL (dtype=object or string). NEVER encode numeric columns (int64, float64) — they are already numbers and encoding them destroys their information.
- onehot: CATEGORICAL columns with 2-5 unique values (creates dummy variables)
- label: CATEGORICAL binary columns or ordinal (natural order)
- target: CATEGORICAL columns with >10 unique values (use frequency encoding)
- features_to_drop: ID columns, names, or columns with >90% nulls
- Do NOT include the target column: "{target_col}"
- ONLY include columns that exist in the dataset
- Return ONLY valid JSON
""",
            expected_output="Pure JSON with encoding strategies and reasoning",
            agent=engineer
        )
        
        crew = Crew(agents=[engineer], tasks=[t])
        result = crew.kickoff()
        
        # Parse LLM strategy
        strategy = parse_json_from_llm(str(result))

        # POST-PROCESSING: Remove numeric columns that the LLM may have
        # mistakenly included in encoding lists. Only categorical (object/string)
        # columns should be encoded.
        if df_dtypes and "encoding_strategy" in strategy:
            enc = strategy["encoding_strategy"]
            numeric_dtypes = {'int64', 'float64', 'int32', 'float32'}
            removed = []
            for enc_type in ['onehot', 'label', 'target']:
                if enc_type in enc and isinstance(enc[enc_type], list):
                    original = enc[enc_type]
                    enc[enc_type] = [
                        col for col in original
                        if col in df_dtypes and df_dtypes[col] not in numeric_dtypes
                    ]
                    removed.extend([c for c in original if c not in enc[enc_type]])
            if removed:
                log(f"[FEATURE] ⚠ Removed {len(removed)} numeric columns from encoding: {removed}")
            strategy["encoding_strategy"] = enc

        # POST-PROCESSING: sanitize features_to_drop (remove placeholders/non-columns)
        features_to_drop = strategy.get("features_to_drop", [])
        if not isinstance(features_to_drop, list):
            features_to_drop = []

        cleaned_drop = []
        for col in features_to_drop:
            if not isinstance(col, str):
                continue
            col_norm = col.strip()
            if col_norm and col_norm in df_dtypes and col_norm != target_col:
                cleaned_drop.append(col_norm)

        # Deduplicate while keeping order
        cleaned_drop = list(dict.fromkeys(cleaned_drop))

        # Safety: avoid dropping almost all features
        if cleaned_drop and df_dtypes and len(cleaned_drop) >= max(1, len(df_dtypes) - 1):
            log("[FEATURE] WARNING: Drop strategy would remove almost all features. Clearing features_to_drop.")
            cleaned_drop = []

        strategy["features_to_drop"] = cleaned_drop

        log(f"[FEATURE] LLM Strategy: {json.dumps(strategy, indent=2)[:500]}")
        
        # If CSV path provided, apply encoding
        encoded_data = None
        if csv_path:
            log(f"[FEATURE] Loading and encoding data...")
            df = pd.read_csv(csv_path)
            original_shape = df.shape
            
            # Drop useless features first
            df = drop_features(df, strategy, target_col, log_callback)
            
            # Apply smart encoding
            df = apply_encoding(df, strategy, target_col, log_callback)
            
            encoded_data = {
                "original_shape": list(original_shape),
                "encoded_shape": list(df.shape),
                "columns": df.columns.tolist()
            }
            log(f"[FEATURE] Done! {original_shape} → {df.shape}")
        
        return A2AResponse(
            task_id=task.task_id,
            sender="feature-agent",
            status="COMPLETED",
            output={
                "feature_strategy": strategy,
                "encoded_data": encoded_data,
                "raw_llm_output": str(result)[:500]
            }
        )
    except Exception as e:
        log(f"[FEATURE] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
