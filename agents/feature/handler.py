from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse
import json
import pandas as pd
import numpy as np

from config import GROQ_MODEL
from core.llm_utils import parse_json_from_llm


def apply_encoding(df: pd.DataFrame, strategy: dict, target_col: str) -> pd.DataFrame:
    """Apply smart encoding based on LLM strategy"""
    print(f"[FEATURE] Applying smart encoding strategies...")
    
    encoding_strategy = strategy.get("encoding_strategy", {})
    
    # One-hot encoding (low cardinality)
    onehot_cols = encoding_strategy.get("onehot", [])
    for col in onehot_cols:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype == 'object':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            print(f"  → {col}: one-hot encoded ({len(dummies.columns)} new columns)")
    
    # Label encoding (ordinal or binary)
    label_cols = encoding_strategy.get("label", [])
    for col in label_cols:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            mapping = {v: i for i, v in enumerate(unique_vals)}
            df[col] = df[col].map(mapping)
            print(f"  → {col}: label encoded ({len(mapping)} classes)")
    
    # Target encoding (high cardinality) - simplified version
    target_cols = encoding_strategy.get("target", [])
    for col in target_cols:
        if col not in df.columns or col == target_col:
            continue
        if df[col].dtype == 'object':
            # For target encoding, use frequency as proxy (safer)
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
            print(f"  → {col}: frequency encoded (proxy for target)")
    
    # Handle any remaining object columns with label encoding
    for col in df.select_dtypes(include=['object']).columns:
        if col == target_col:
            continue
        unique_vals = df[col].unique()
        mapping = {v: i for i, v in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)
        print(f"  → {col}: auto label encoded ({len(mapping)} classes)")
    
    return df


def drop_features(df: pd.DataFrame, strategy: dict, target_col: str) -> pd.DataFrame:
    """Drop features recommended by LLM"""
    features_to_drop = strategy.get("features_to_drop", [])
    
    for col in features_to_drop:
        if col in df.columns and col != target_col:
            df = df.drop(col, axis=1)
            print(f"  → Dropped: {col}")
    
    return df


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle(task: A2ATask):
    try:
        # Get analysis info
        analysis = task.input.get("analysis_summary", task.input)
        csv_path = task.input.get("csv_path")
        target_col = analysis.get("target_column", "")
        cardinality = analysis.get("cardinality", {})
        
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
    "onehot": ["low_cardinality_cols"],
    "label": ["binary_or_ordinal_cols"],
    "target": ["high_cardinality_cols"],
    "reason": "brief explanation of why each column was assigned that encoding type"
  }},
  "features_to_drop": ["id_cols", "name_cols", "useless_cols"],
  "drop_reason": "brief explanation of why these features should be dropped"
}}

Rules:
- onehot: Columns with 2-5 unique values (creates dummy variables)
- label: Binary columns or ordinal (natural order) 
- target: Columns with >10 unique values (use frequency encoding)
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
        print(f"[FEATURE] LLM Strategy: {json.dumps(strategy, indent=2)[:500]}")
        
        # If CSV path provided, apply encoding
        encoded_data = None
        if csv_path:
            print(f"[FEATURE] Loading and encoding data...")
            df = pd.read_csv(csv_path)
            original_shape = df.shape
            
            # Drop useless features first
            df = drop_features(df, strategy, target_col)
            
            # Apply smart encoding
            df = apply_encoding(df, strategy, target_col)
            
            encoded_data = {
                "original_shape": list(original_shape),
                "encoded_shape": list(df.shape),
                "columns": df.columns.tolist()
            }
            print(f"[FEATURE] Done! {original_shape} → {df.shape}")
        
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
        print(f"[FEATURE] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
