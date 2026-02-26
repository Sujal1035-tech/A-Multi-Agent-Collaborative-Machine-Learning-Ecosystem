"""
Shared Data Utilities
Common data-cleaning functions used by multiple agents — extracted to avoid duplication.
"""

import pandas as pd

MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "?", "missing"}


def normalize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common string placeholders to actual NaN."""
    df = df.copy()
    text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    for col in text_cols:
        normalized = df[col].astype("string").str.strip()
        mask = normalized.str.lower().isin(MISSING_TOKENS)
        if mask.any():
            df.loc[mask, col] = pd.NA
    return df

def load_csv_robust(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Robust CSV loader that tries multiple encodings and separators.
    Falls back to python engine and skips bad lines if standard parsing fails.
    """
    import os
    if not os.path.exists(filepath):
         raise FileNotFoundError(f"File not found: {filepath}")

    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'utf-16']
    separators_to_try = [',', ';', '\t', '|']

    # 1. Try standard reading first (fastest)
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(filepath, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    # 2. Try sniffing separator with different encodings
    for encoding in encodings_to_try:
        for sep in separators_to_try:
            try:
                # Try reading a small chunk to see if separating works better
                df = pd.read_csv(filepath, encoding=encoding, sep=sep, on_bad_lines='skip', engine='python', **kwargs)
                if len(df.columns) > 1 or sep == ',': # Success criteria: more than 1 column found
                    return df
            except Exception:
                continue
    
    # 3. Last resort: Ignore errors completely on read
    try:
        return pd.read_csv(filepath, encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip', engine='python', **kwargs)
    except Exception as e:
        raise ValueError(f"Could not parse CSV {filepath} with any strategy. Last error: {str(e)}")

