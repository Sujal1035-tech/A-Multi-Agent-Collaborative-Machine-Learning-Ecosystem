from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse
import json
import pandas as pd
import numpy as np

from config import GEMINI_MODEL
from core.llm_utils import parse_json_from_llm
from core.data_utils import load_csv_robust


# =============================================================================
# FIT / TRANSFORM — Feature Dropping
# =============================================================================

def fit_drop_features(X, strategy, target_col, log_callback=None):
    """
    Determine which features to drop based on LLM strategy (fit on train).
    Returns (X, state) where state stores the list of dropped columns.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    state = {"dropped_cols": []}
    features_to_drop = strategy.get("features_to_drop", strategy.get("feature_strategy", {}).get("features_to_drop", []))
    if not isinstance(features_to_drop, list):
        features_to_drop = []

    existing_drop = [c for c in features_to_drop if isinstance(c, str) and c in X.columns and c != target_col]

    # SAFETY GUARD: Never drop more than 50% of columns
    max_drop = max(1, len(X.columns) // 2)
    if len(existing_drop) > max_drop:
        log(f"[FEATURE] WARNING: LLM wanted to drop {len(existing_drop)} columns but max allowed is {max_drop}.")
        safe_drop = [c for c in existing_drop if X[c].dtype == 'object' or
                     'id' in c.lower() or 'name' in c.lower() or 'unnamed' in c.lower()]
        existing_drop = safe_drop[:max_drop]

    # Never allow dropping all features
    if len(existing_drop) >= len(X.columns):
        log("[FEATURE] WARNING: Drop strategy would remove all features. Ignoring.")
        existing_drop = []

    if existing_drop:
        X = X.drop(columns=existing_drop)
        state["dropped_cols"] = existing_drop
        log(f"[FEATURE] Dropped features: {existing_drop}")

    return X, state


def transform_drop_features(X, state, log_callback=None):
    """Drop the same features from test data as were dropped from train."""
    existing_drop = [c for c in state["dropped_cols"] if c in X.columns]
    if existing_drop:
        X = X.drop(columns=existing_drop)
    return X


# =============================================================================
# FIT / TRANSFORM — Datetime Extraction
# =============================================================================

def fit_datetime(X, strategy, target_col, log_callback=None):
    """
    Detect and extract datetime features. (Fit stores which columns are datetimes).
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    state = {"datetime_cols": []}
    dt_strategy = strategy.get("datetime_strategy", strategy.get("feature_strategy", {}).get("datetime_strategy", {}))
    dt_cols = dt_strategy.get("columns", [])

    for col in dt_cols:
        if col in X.columns and col != target_col:
            # Try parsing to datetime
            try:
                dt_series = pd.to_datetime(X[col], errors='coerce')
                if dt_series.notna().sum() > 0: # If at least some parsed successfully
                    state["datetime_cols"].append(col)
                    
                    # Extract features
                    X[f"{col}_year"] = dt_series.dt.year.fillna(0)
                    X[f"{col}_month"] = dt_series.dt.month.fillna(0)
                    X[f"{col}_day"] = dt_series.dt.day.fillna(0)
                    X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0)
                    X[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
                    
                    # Drop original string column
                    X = X.drop(columns=[col])
                    log(f"[FEATURE] Extracted datetime features from: {col}")
            except Exception as e:
                log(f"[FEATURE] WARNING: Failed to parse datetime {col}: {e}")

    return X, state

def transform_datetime(X, state, log_callback=None):
    """Apply datetime extraction to test data."""
    for col in state["datetime_cols"]:
        if col in X.columns:
            try:
                dt_series = pd.to_datetime(X[col], errors='coerce')
                X[f"{col}_year"] = dt_series.dt.year.fillna(0)
                X[f"{col}_month"] = dt_series.dt.month.fillna(0)
                X[f"{col}_day"] = dt_series.dt.day.fillna(0)
                X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0)
                X[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
                X = X.drop(columns=[col])
            except Exception:
                pass
    return X

# =============================================================================
# FIT / TRANSFORM — Text Vectorization (TF-IDF)
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

def fit_text_vectorization(X, strategy, target_col, log_callback=None):
    """
    Extract TF-IDF vectors from high-cardinality/long text columns.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    state = {"text_cols": {}}
    text_strategy = strategy.get("text_strategy", strategy.get("feature_strategy", {}).get("text_strategy", {}))
    text_cols = text_strategy.get("columns", [])

    for col in text_cols:
        if col in X.columns and col != target_col and X[col].dtype == 'object':
            X[col] = X[col].fillna("").astype(str)
            # Only apply if average length is somewhat long (preventing this on simple labels)
            avg_len = X[col].str.len().mean()
            if avg_len > 15:
                # Use a small max_features to avoid exploding the dataset width
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                try:
                    tfidf_matrix = vectorizer.fit_transform(X[col])
                    # Create DataFrame with new columns
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(), 
                        columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                        index=X.index
                    )
                    state["text_cols"][col] = vectorizer
                    X = pd.concat([X.drop(col, axis=1), tfidf_df], axis=1)
                    log(f"[FEATURE] Vectorized text column: {col} -> {tfidf_matrix.shape[1]} TF-IDF features")
                except Exception as e:
                    log(f"[FEATURE] WARNING: Failed to vectorize text {col}: {e}")
            else:
                 log(f"[FEATURE] Text column {col} too short (avg len {avg_len:.1f}), skipping TF-IDF")

    return X, state

def transform_text_vectorization(X, state, log_callback=None):
    """Apply TF-IDF extraction to test data."""
    for col, vectorizer in state["text_cols"].items():
        if col in X.columns:
            X[col] = X[col].fillna("").astype(str)
            try:
                tfidf_matrix = vectorizer.transform(X[col])
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                    index=X.index
                )
                X = pd.concat([X.drop(col, axis=1), tfidf_df], axis=1)
            except Exception:
                # Add zero columns if failed
                features = [f"{col}_tfidf_{i}" for i in range(len(vectorizer.get_feature_names_out()))]
                for f in features:
                   X[f] = 0.0
                X = X.drop(columns=[col])
    return X

# =============================================================================
# FIT / TRANSFORM — Encoding
# =============================================================================

def fit_encoding(X, strategy, target_col, log_callback=None):
    """
    Fit encoding on TRAINING data only.
    Returns (X, state) where state stores encoding mappings.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    state = {"onehot_cols": [], "label_maps": {}, "freq_maps": {}}
    encoding = strategy.get("encoding_strategy", strategy.get("feature_strategy", {}).get("encoding_strategy", {}))

    # One-hot encoding (low cardinality)
    onehot_cols = encoding.get("onehot", [])
    for col in onehot_cols:
        if col in X.columns and X[col].dtype == 'object':
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            state["onehot_cols"].append({"col": col, "categories": list(dummies.columns)})
            X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
            log(f"[FEATURE] One-hot: {col} ({len(dummies.columns)} cols)")

    # Label encoding (ordinal or binary)
    label_cols = encoding.get("label", [])
    for col in label_cols:
        if col in X.columns and X[col].dtype == 'object':
            unique_vals = X[col].unique()
            mapping = {v: i for i, v in enumerate(unique_vals)}
            state["label_maps"][col] = mapping
            X[col] = X[col].map(mapping)
            log(f"[FEATURE] Label encode: {col} ({len(mapping)} classes)")

    # Frequency encoding (safe proxy for target encoding)
    target_encode_cols = encoding.get("target", [])
    for col in target_encode_cols:
        if col in X.columns and X[col].dtype == 'object':
            freq = X[col].value_counts(normalize=True).to_dict()
            state["freq_maps"][col] = freq
            X[col] = X[col].map(freq).fillna(0)
            log(f"[FEATURE] Frequency encode: {col}")

    # Fallback: encode remaining object columns
    for col in X.select_dtypes(include=['object']).columns:
        if col == target_col:
            continue
        unique_vals = X[col].unique()
        mapping = {v: i for i, v in enumerate(unique_vals)}
        state["label_maps"][col] = mapping
        X[col] = X[col].map(mapping)
        log(f"[FEATURE] Auto-encode: {col}")

    return X, state


def transform_encoding(X, state, log_callback=None):
    """Apply encoding to test data using train-fitted mappings."""
    # One-hot encoding (align to train categories)
    for info in state["onehot_cols"]:
        col = info["col"]
        categories = info["categories"]
        if col in X.columns and X[col].dtype == 'object':
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            for cat in categories:
                if cat not in dummies.columns:
                    dummies[cat] = 0
            dummies = dummies[[c for c in categories if c in dummies.columns]]
            X = pd.concat([X.drop(col, axis=1), dummies], axis=1)

    # Label encoding (use train mapping)
    for col, mapping in state["label_maps"].items():
        if col in X.columns and X[col].dtype == 'object':
            X[col] = X[col].map(mapping).fillna(-1).astype(int)

    # Frequency encoding (use train frequencies)
    for col, freq in state["freq_maps"].items():
        if col in X.columns and X[col].dtype == 'object':
            X[col] = X[col].map(freq).fillna(0)

    return X


# =============================================================================
# MAIN HANDLER — LLM Strategy Generation (no data application)
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
                _df = load_csv_robust(csv_path, nrows=5)
                df_dtypes = {col: str(_df[col].dtype) for col in _df.columns}
            except Exception:
                pass

        engineer = Agent(
            role="Smart Feature Engineering Expert",
            goal="Analyze data and return ONLY valid JSON with encoding strategies",
            backstory="""You analyze data cardinality and types to recommend optimal temporal, text, and categorical feature engineering.
            - Datetime columns (Dates/Timestamps) -> datetime_strategy (extracts year/month/day)
            - Long Text columns (Descriptions/Transcripts >20 chars avg) -> text_strategy (TF-IDF vectorization)
            - Categorical One-hot for low cardinality (<5 unique values)
            - Categorical Label for binary or ordinal data
            - Categorical Target/frequency encoding for high cardinality (>10 unique)
            Always return ONLY a valid JSON object.""",
            llm=GEMINI_MODEL
        )

        # Truncate input
        input_str = str(analysis)
        if len(input_str) > 4000:
            input_str = input_str[:4000] + "..."

        t = Task(
            description=f"""
Analyze these data stats and return optimal feature engineering strategies as JSON:

{input_str}

Return ONLY this exact JSON structure:
{{
  "datetime_strategy": {{
    "columns": ["transaction_date", "created_at"],
    "reason": "These look like dates/timestamps"
  }},
  "text_strategy": {{
    "columns": ["description", "transcript", "notes"],
    "reason": "These look like long free-text columns"
  }},
  "encoding_strategy": {{
    "onehot": ["short_categorical_cols_less_than_5_unique"],
    "label": ["binary_cols"],
    "target": ["high_card_categorical_cols"],
    "reason": "explanation of encoding choices"
  }},
  "features_to_drop": ["id_cols", "pure_noise"],
  "drop_reason": "why dropping"
}}

Rules:
- CRITICAL: Do NOT mix types. If a column is in datetime_strategy, do NOT put it in encoding_strategy.
- datetime: ANY column that looks like a date, timestamp, or time (e.g. YYYY-MM-DD). We will extract year/month/day.
- text: ANY object/string column that is LONG free-text (e.g. transcripts, reviews, descriptions). We will apply NLP TF-IDF vectorization.
- encoding: ONLY encode CATEGORICAL columns (dtype=object or string). NEVER encode numeric columns.
- features_to_drop: ID columns or columns with >90% nulls. Do NOT drop dates or text, we can extract features from them!
- Do NOT include the target column: "{target_col}"
- ONLY include columns that exist in the dataset
- Return ONLY valid JSON
""",
            expected_output="Pure JSON with encoding strategies and reasoning",
            agent=engineer
        )

        crew = Crew(agents=[engineer], tasks=[t])
        result = crew.kickoff()

        # Phase 1: Robustly extract JSON from messy LLM output (strips markdown, etc.)
        strategy = parse_json_from_llm(str(result))
        
        # Flatten nested 'feature_strategy' common LLM hallucination
        if "feature_strategy" in strategy and "encoding_strategy" in strategy["feature_strategy"]:
            strategy = strategy["feature_strategy"]
            
        # Ensure default keys to prevent KeyError down pipeline
        strategy.setdefault("datetime_strategy", {"columns": [], "reason": ""})
        strategy.setdefault("text_strategy", {"columns": [], "reason": ""})
        strategy.setdefault("encoding_strategy", {"onehot": [], "label": [], "target": [], "reason": ""})
        strategy.setdefault("features_to_drop", [])

        # POST-PROCESSING: Remove numeric columns that the LLM may have
        # mistakenly included in encoding lists.
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

        # POST-PROCESSING: sanitize features_to_drop
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

        cleaned_drop = list(dict.fromkeys(cleaned_drop))

        if cleaned_drop and df_dtypes and len(cleaned_drop) >= max(1, len(df_dtypes) - 1):
            log("[FEATURE] WARNING: Drop strategy would remove almost all features. Clearing features_to_drop.")
            cleaned_drop = []

        strategy["features_to_drop"] = cleaned_drop

        log(f"[FEATURE] LLM Strategy: {json.dumps(strategy, indent=2)[:500]}")

        # Strategy-only output: actual application is done by the model agent
        # with proper fit/transform separation to prevent data leakage.
        return A2AResponse(
            task_id=task.task_id,
            sender="feature-agent",
            status="COMPLETED",
            output={
                "feature_strategy": strategy,
                "raw_llm_output": str(result)[:500]
            }
        )
    except Exception as e:
        log(f"[FEATURE] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
