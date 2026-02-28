"""
Shared Preprocessing Step Definitions
Single source of truth for preprocessing logic used by both:
  - agents/model/handler.py (runtime execution)
  - agents/project/handler.py (code generation)

This prevents the two files from going out of sync.
"""

# =============================================================================
# CONSTANTS — shared between runtime and code generation
# =============================================================================

MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "?", "missing"}

DEFAULT_OUTLIER_METHOD = "iqr_capping"
DEFAULT_OUTLIER_THRESHOLD = 1.5

DEFAULT_SCALING_METHOD = "robust"

NUMERIC_DTYPES = {"int64", "float64", "int32", "float32"}


# =============================================================================
# CODE GENERATION — used by project/handler.py
# =============================================================================

def generate_null_handling_code(null_strategy: dict) -> str:
    """Generate Python code string for null handling."""
    if not null_strategy:
        return ""

    code = "\n# --- Null handling ---\n"
    for col, details in null_strategy.items():
        method = details.get("method", "median") if isinstance(details, dict) else details

        if method in ("mean", "median"):
            code += f"if '{col}' in X_train.columns and X_train['{col}'].dtype in ['int64', 'float64']:\n"
            code += f"    _fill = X_train['{col}'].{method}()  # Computed from train only\n"
            code += f"    X_train['{col}'] = X_train['{col}'].fillna(_fill)\n"
            code += f"    X_test['{col}'] = X_test['{col}'].fillna(_fill)\n"
        elif method == "mode":
            code += f"if '{col}' in X_train.columns:\n"
            code += f"    _fill = X_train['{col}'].mode()[0] if len(X_train['{col}'].mode()) > 0 else 'Unknown'\n"
            code += f"    X_train['{col}'] = X_train['{col}'].fillna(_fill)\n"
            code += f"    X_test['{col}'] = X_test['{col}'].fillna(_fill)\n"
        elif method == "knn":
            # KNN removed - fall back to median
            code += f"if '{col}' in X_train.columns and X_train['{col}'].dtype in ['int64', 'float64']:\n"
            code += f"    _fill = X_train['{col}'].median()  # Computed from train only\n"
            code += f"    X_train['{col}'] = X_train['{col}'].fillna(_fill)\n"
            code += f"    X_test['{col}'] = X_test['{col}'].fillna(_fill)\n"
        elif method == "drop":
            code += f"if '{col}' in X_train.columns:\n"
            code += f"    _mask = X_train['{col}'].notna()\n"
            code += f"    X_train = X_train[_mask]\n"
            code += f"    y_train = y_train[_mask]\n"

    return code


def generate_drop_features_code(features_to_drop: list) -> str:
    """Generate Python code string for dropping features."""
    if not features_to_drop:
        return ""

    code = f"\n# --- Drop unused features ---\n"
    code += f"_drop_cols = [c for c in {features_to_drop} if c in X_train.columns]\n"
    code += f"X_train = X_train.drop(columns=_drop_cols, errors='ignore')\n"
    code += f"X_test = X_test.drop(columns=_drop_cols, errors='ignore')\n"
    return code


def generate_outlier_code(method: str, columns: list, threshold: float) -> str:
    """Generate Python code string for outlier handling."""
    if method != "iqr_capping" or not columns:
        return ""

    code = f"\n# --- Outlier handling (IQR bounds from train only) ---\n"
    code += f"for col in {columns}:\n"
    code += f"    if col in X_train.columns and X_train[col].dtype in ['int64', 'float64']:\n"
    code += f"        Q1 = X_train[col].quantile(0.25)\n"
    code += f"        Q3 = X_train[col].quantile(0.75)\n"
    code += f"        IQR = Q3 - Q1\n"
    code += f"        lower = Q1 - {threshold} * IQR\n"
    code += f"        upper = Q3 + {threshold} * IQR\n"
    code += f"        X_train[col] = X_train[col].clip(lower=lower, upper=upper)\n"
    code += f"        X_test[col] = X_test[col].clip(lower=lower, upper=upper)  # Same bounds from train\n"
    return code


def generate_encoding_code(encoding_strategy: dict) -> str:
    """Generate Python code string for all encoding steps."""
    code = "\n# --- Encoding ---\n"

    onehot_cols = encoding_strategy.get("onehot", [])
    if onehot_cols:
        code += f"# One-Hot Encoding (aligned to training categories)\n"
        code += f"_onehot_cols = [c for c in {onehot_cols} if c in X_train.columns and X_train[c].dtype == 'object']\n"
        code += "for col in _onehot_cols:\n"
        code += "    _train_dummies = pd.get_dummies(X_train[col], prefix=col, drop_first=True)\n"
        code += "    _test_dummies = pd.get_dummies(X_test[col], prefix=col, drop_first=True)\n"
        code += "    # Align test to train columns\n"
        code += "    for c in _train_dummies.columns:\n"
        code += "        if c not in _test_dummies.columns:\n"
        code += "            _test_dummies[c] = 0\n"
        code += "    _test_dummies = _test_dummies[[c for c in _train_dummies.columns if c in _test_dummies.columns]]\n"
        code += "    X_train = pd.concat([X_train.drop(col, axis=1), _train_dummies], axis=1)\n"
        code += "    X_test = pd.concat([X_test.drop(col, axis=1), _test_dummies], axis=1)\n\n"

    label_cols = encoding_strategy.get("label", [])
    if label_cols:
        code += f"# Label Encoding (fit on train, transform both)\n"
        code += f"_label_cols = [c for c in {label_cols} if c in X_train.columns and X_train[c].dtype == 'object']\n"
        code += "for col in _label_cols:\n"
        code += "    _le = LabelEncoder()\n"
        code += "    X_train[col] = _le.fit_transform(X_train[col].astype(str))\n"
        code += "    X_test[col] = X_test[col].map({v: i for i, v in enumerate(_le.classes_)})\n"
        code += "    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1).astype(int)\n\n"

    target_enc_cols = encoding_strategy.get("target", [])
    if target_enc_cols:
        code += f"# Frequency Encoding (frequencies from train only)\n"
        code += f"_freq_cols = [c for c in {target_enc_cols} if c in X_train.columns and X_train[c].dtype == 'object']\n"
        code += "for col in _freq_cols:\n"
        code += "    _freq = X_train[col].value_counts(normalize=True).to_dict()\n"
        code += "    X_train[col] = X_train[col].map(_freq).fillna(0)\n"
        code += "    X_test[col] = X_test[col].map(_freq).fillna(0)  # Same frequencies from train\n\n"

    # Auto-encode remaining categorical columns
    code += "# Auto-encode remaining categorical columns\n"
    code += "for col in X_train.select_dtypes(include=['object', 'category', 'string']).columns:\n"
    code += "    _le = LabelEncoder()\n"
    code += "    X_train[col] = _le.fit_transform(X_train[col].astype(str))\n"
    code += "    _mapping = {v: i for i, v in enumerate(_le.classes_)}\n"
    code += "    X_test[col] = X_test[col].map(_mapping)\n"
    code += "    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1).astype(int)\n\n"

    return code


def generate_safety_net_code() -> str:
    """Generate Python code for final numeric safety net + NaN filling."""
    code = "# Final Safety Net: Force all features to be pure numeric to prevent XGBoost type crashes\n"
    code += "for col in X_train.columns:\n"
    code += "    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')\n"
    code += "    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')\n\n"

    code += "# Fill any remaining NaN (using train medians)\n"
    code += "if X_train.isna().sum().sum() > 0:\n"
    code += "    _medians = X_train.median(numeric_only=True)\n"
    code += "    X_train = X_train.fillna(_medians)\n"
    code += "    X_test = X_test.fillna(_medians)\n\n"

    code += "\n# Align test columns to match training\n"
    code += "for col in X_train.columns:\n"
    code += "    if col not in X_test.columns:\n"
    code += "        X_test[col] = 0\n"
    code += "X_test = X_test[X_train.columns]\n"

    return code
