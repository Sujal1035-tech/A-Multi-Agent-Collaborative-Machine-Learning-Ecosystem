"""
Project Agent Handler
Generates deployable analysis.py code that EXACTLY mirrors the pipeline decisions.

The generated code reproduces the same preprocessing, encoding, model training,
and evaluation steps that the agents performed during the pipeline — including
the split-before-preprocess pattern to prevent data leakage.
"""

from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse

from config import GROQ_MODEL


def handle_project(task: A2ATask, log_callback=None) -> A2AResponse:
    """Handle project generation task"""
    def log(msg):
        if log_callback:
            log_callback(msg)

    try:
        log("[PROJECT] Starting project code generation (Deterministic)...")

        # Extract all pipeline decisions
        best_model_info = task.input.get('best_model_info', {})
        best_name = best_model_info.get('model', 'Unknown')
        best_params = best_model_info.get('params', {})
        used_balancing = best_model_info.get('used_balancing', False)
        used_scaling = best_model_info.get('used_scaling', True)
        target_transform = best_model_info.get('target_transform', False)
        problem_type = best_model_info.get('problem_type', 'classification')
        target_column = task.input.get('target_column', None)

        prep_strategy = task.input.get('prep_strategy', {})
        if "preprocessing_strategy" in prep_strategy:
            prep_strategy = prep_strategy["preprocessing_strategy"]

        feat_strategy = task.input.get('feat_strategy', {})
        if "feature_strategy" in feat_strategy:
            feat_strategy = feat_strategy["feature_strategy"]

        null_strategy = prep_strategy.get('null_strategy', {})
        encoding_strategy = feat_strategy.get('encoding_strategy', {})
        features_to_drop = feat_strategy.get('features_to_drop', [])
        outlier_strategy = prep_strategy.get('outlier_strategy', {})
        outlier_method = outlier_strategy.get("method", "iqr_capping")
        outlier_cols = outlier_strategy.get("columns", [])
        outlier_threshold = outlier_strategy.get("threshold", 1.5)

        # Clean params
        best_params = dict(best_params)  # Copy
        best_params.pop('model', None)
        if 'random_state' not in best_params:
            best_params['random_state'] = 42

        is_classification = (problem_type == 'classification')

        # =====================================================================
        # ASSEMBLE analysis.py CODE
        # This code mirrors EXACTLY what model/handler.py does
        # =====================================================================

        code = '"""'\
            '\nAutoEDA Generated Analysis\n'\
            'This code reproduces the exact pipeline that the agents performed.\n'\
            f'Best Model: {best_name}\n'\
            f'Problem Type: {problem_type}\n'\
            '"""\n\n'

        # --- Imports ---
        code += "import pandas as pd\n"
        code += "import numpy as np\n"
        code += "import matplotlib.pyplot as plt\n"
        code += "import seaborn as sns\n"
        code += "import os\n"
        code += "import warnings\n"
        code += "warnings.filterwarnings('ignore')\n\n"

        code += "from sklearn.model_selection import train_test_split\n"
        if is_classification:
            code += "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
        else:
            code += "from sklearn.model_selection import cross_val_score\n"
        code += "from sklearn.preprocessing import LabelEncoder, StandardScaler\n"

        if target_transform:
            code += "from sklearn.preprocessing import PowerTransformer\n"
        if used_balancing:
            code += "try:\n    from imblearn.over_sampling import SMOTE\nexcept ImportError:\n    SMOTE = None\n"

        # Metric imports
        if is_classification:
            code += "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
            code += "from sklearn.metrics import precision_score, recall_score, f1_score\n"
        else:
            code += "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n"

        # Model imports — only import what's needed based on best model
        code += "\n# Model Import\n"
        model_import = _get_model_import(best_name, problem_type)
        code += model_import + "\n"

        # --- Setup ---
        code += "\n# ============================================\n"
        code += "# 1. SETUP\n"
        code += "# ============================================\n"
        code += "os.makedirs('stats', exist_ok=True)\n"
        code += "os.makedirs('plots', exist_ok=True)\n"
        code += "os.makedirs('reports', exist_ok=True)\n"

        # --- Load Data ---
        code += "\n# ============================================\n"
        code += "# 2. LOAD DATA\n"
        code += "# ============================================\n"
        code += "try:\n"
        code += "    data = pd.read_csv('data.csv')\n"
        code += "    print(f'Data loaded. Shape: {data.shape}')\n"
        code += "except Exception as e:\n"
        code += "    print(f'Error loading data: {e}')\n"
        code += "    exit(1)\n"

        # --- Separate X/y using the ACTUAL target column ---
        code += "\n# ============================================\n"
        code += "# 3. SEPARATE FEATURES AND TARGET\n"
        code += "# ============================================\n"
        if target_column:
            code += f"TARGET_COL = '{target_column}'\n"
        else:
            code += "TARGET_COL = data.columns[-1]  # Fallback to last column\n"
        code += "X = data.drop(TARGET_COL, axis=1)\n"
        code += "y = data[TARGET_COL]\n\n"

        # Encode target if needed
        code += "# Encode target if categorical\n"
        code += "target_le = None\n"
        code += "if y.dtype == 'object' or str(y.dtype) == 'category':\n"
        code += "    target_le = LabelEncoder()\n"
        code += "    y = pd.Series(target_le.fit_transform(y), index=y.index)\n"
        code += "    print(f'Encoded target: {dict(zip(target_le.classes_, range(len(target_le.classes_))))}')\n"

        # --- SPLIT FIRST (before any preprocessing) ---
        code += "\n# ============================================\n"
        code += "# 4. TRAIN/TEST SPLIT (BEFORE preprocessing to prevent data leakage)\n"
        code += "# ============================================\n"
        if is_classification:
            code += "X_train, X_test, y_train, y_test = train_test_split(\n"
            code += "    X, y, test_size=0.2, random_state=42, stratify=y  # Stratified for classification\n"
            code += ")\n"
        else:
            code += "X_train, X_test, y_train, y_test = train_test_split(\n"
            code += "    X, y, test_size=0.2, random_state=42\n"
            code += ")\n"
        code += "print(f'Train: {X_train.shape}, Test: {X_test.shape}')\n"

        # --- PREPROCESSING (fit on train, transform both) ---
        code += "\n# ============================================\n"
        code += "# 5. PREPROCESSING (fit on TRAIN only, transform both)\n"
        code += "#    This prevents data leakage from test set\n"
        code += "# ============================================\n"

        # Null handling (fit on train)
        if null_strategy:
            code += "\n# --- Null handling ---\n"
            for col, details in null_strategy.items():
                if isinstance(details, dict):
                    method = details.get('method', 'median')
                else:
                    method = details

                if method == 'mean':
                    code += f"if '{col}' in X_train.columns and X_train['{col}'].dtype in ['int64', 'float64']:\n"
                    code += f"    _fill = X_train['{col}'].mean()  # Computed from train only\n"
                    code += f"    X_train['{col}'] = X_train['{col}'].fillna(_fill)\n"
                    code += f"    X_test['{col}'] = X_test['{col}'].fillna(_fill)\n"
                elif method == 'median':
                    code += f"if '{col}' in X_train.columns and X_train['{col}'].dtype in ['int64', 'float64']:\n"
                    code += f"    _fill = X_train['{col}'].median()  # Computed from train only\n"
                    code += f"    X_train['{col}'] = X_train['{col}'].fillna(_fill)\n"
                    code += f"    X_test['{col}'] = X_test['{col}'].fillna(_fill)\n"
                elif method == 'mode':
                    code += f"if '{col}' in X_train.columns:\n"
                    code += f"    _fill = X_train['{col}'].mode()[0] if len(X_train['{col}'].mode()) > 0 else 'Unknown'\n"
                    code += f"    X_train['{col}'] = X_train['{col}'].fillna(_fill)\n"
                    code += f"    X_test['{col}'] = X_test['{col}'].fillna(_fill)\n"
                elif method == 'knn':
                    code += f"if '{col}' in X_train.columns and X_train['{col}'].dtype in ['int64', 'float64']:\n"
                    code += f"    try:\n"
                    code += f"        from sklearn.impute import KNNImputer\n"
                    code += f"        _imputer = KNNImputer(n_neighbors=5)\n"
                    code += f"        X_train[['{col}']] = _imputer.fit_transform(X_train[['{col}']])\n"
                    code += f"        X_test[['{col}']] = _imputer.transform(X_test[['{col}']])\n"
                    code += f"    except ImportError:\n"
                    code += f"        _fill = X_train['{col}'].median()\n"
                    code += f"        X_train['{col}'] = X_train['{col}'].fillna(_fill)\n"
                    code += f"        X_test['{col}'] = X_test['{col}'].fillna(_fill)\n"
                elif method == 'drop':
                    code += f"if '{col}' in X_train.columns:\n"
                    code += f"    _mask = X_train['{col}'].notna()\n"
                    code += f"    X_train = X_train[_mask]\n"
                    code += f"    y_train = y_train[_mask]\n"

        # Drop features
        if features_to_drop:
            code += f"\n# --- Drop unused features ---\n"
            code += f"_drop_cols = [c for c in {features_to_drop} if c in X_train.columns]\n"
            code += f"X_train = X_train.drop(columns=_drop_cols, errors='ignore')\n"
            code += f"X_test = X_test.drop(columns=_drop_cols, errors='ignore')\n"

        # Outlier handling (bounds from train)
        if outlier_method == 'iqr_capping' and outlier_cols:
            code += f"\n# --- Outlier handling (IQR bounds from train only) ---\n"
            code += f"for col in {outlier_cols}:\n"
            code += f"    if col in X_train.columns and X_train[col].dtype in ['int64', 'float64']:\n"
            code += f"        Q1 = X_train[col].quantile(0.25)\n"
            code += f"        Q3 = X_train[col].quantile(0.75)\n"
            code += f"        IQR = Q3 - Q1\n"
            code += f"        lower = Q1 - {outlier_threshold} * IQR\n"
            code += f"        upper = Q3 + {outlier_threshold} * IQR\n"
            code += f"        X_train[col] = X_train[col].clip(lower=lower, upper=upper)\n"
            code += f"        X_test[col] = X_test[col].clip(lower=lower, upper=upper)  # Same bounds from train\n"

        # Encoding (fit on train)
        code += "\n# --- Encoding ---\n"

        onehot_cols = encoding_strategy.get('onehot', [])
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

        label_cols = encoding_strategy.get('label', [])
        if label_cols:
            code += f"# Label Encoding (fit on train, transform both)\n"
            code += f"_label_cols = [c for c in {label_cols} if c in X_train.columns and X_train[c].dtype == 'object']\n"
            code += "for col in _label_cols:\n"
            code += "    _le = LabelEncoder()\n"
            code += "    X_train[col] = _le.fit_transform(X_train[col].astype(str))\n"
            code += "    X_test[col] = X_test[col].map({v: i for i, v in enumerate(_le.classes_)})\n"
            code += "    X_test[col] = X_test[col].fillna(-1).astype(int)\n\n"

        target_enc_cols = encoding_strategy.get('target', [])
        if target_enc_cols:
            code += f"# Frequency Encoding (frequencies from train only)\n"
            code += f"_freq_cols = [c for c in {target_enc_cols} if c in X_train.columns and X_train[c].dtype == 'object']\n"
            code += "for col in _freq_cols:\n"
            code += "    _freq = X_train[col].value_counts(normalize=True).to_dict()\n"
            code += "    X_train[col] = X_train[col].map(_freq).fillna(0)\n"
            code += "    X_test[col] = X_test[col].map(_freq).fillna(0)  # Same frequencies from train\n\n"

        # Fallback encoding
        code += "# Auto-encode remaining categorical columns\n"
        code += "for col in X_train.select_dtypes(include=['object', 'category']).columns:\n"
        code += "    _le = LabelEncoder()\n"
        code += "    X_train[col] = _le.fit_transform(X_train[col].astype(str))\n"
        code += "    _mapping = {v: i for i, v in enumerate(_le.classes_)}\n"
        code += "    X_test[col] = X_test[col].map(_mapping).fillna(-1).astype(int)\n"

        # Fallback NaN fill
        code += "\n# Fill any remaining NaN (using train medians)\n"
        code += "if X_train.isna().sum().sum() > 0:\n"
        code += "    _medians = X_train.median(numeric_only=True)\n"
        code += "    X_train = X_train.fillna(_medians)\n"
        code += "    X_test = X_test.fillna(_medians)\n"

        # Align test columns to train
        code += "\n# Align test columns to match training\n"
        code += "for col in X_train.columns:\n"
        code += "    if col not in X_test.columns:\n"
        code += "        X_test[col] = 0\n"
        code += "X_test = X_test[X_train.columns]\n"

        # --- Advanced Preprocessing ---
        code += "\n# ============================================\n"
        code += "# 6. ADVANCED PREPROCESSING\n"
        code += "# ============================================\n"

        if used_balancing:
            code += "\n# Class Imbalance Handling (SMOTE on training data only)\n"
            code += "if SMOTE is not None:\n"
            code += "    _vc = y_train.value_counts()\n"
            code += "    _ratio = _vc.min() / _vc.max()\n"
            code += "    if _ratio < 0.5:  # Only apply if actually imbalanced\n"
            code += "        try:\n"
            code += "            smote = SMOTE(random_state=42)\n"
            code += "            X_train, y_train = smote.fit_resample(X_train, y_train)\n"
            code += "            X_train = pd.DataFrame(X_train, columns=X_test.columns)\n"
            code += "            y_train = pd.Series(y_train)\n"
            code += "            print(f'SMOTE applied: {len(X_train)} samples')\n"
            code += "        except Exception as e:\n"
            code += "            print(f'SMOTE failed: {e}')\n"

        if used_scaling:
            code += "\n# Feature Scaling\n"
            code += "scaler = StandardScaler()\n"
            code += "X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)\n"
            code += "X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)\n"

        if target_transform:
            code += "\n# Target Transform (Yeo-Johnson for skewed regression target)\n"
            code += "pt = PowerTransformer(method='yeo-johnson')\n"
            code += "y_train = pt.fit_transform(y_train.values.reshape(-1, 1)).ravel()\n"

        # --- Model Training ---
        code += "\n# ============================================\n"
        code += f"# 7. MODEL TRAINING — {best_name}\n"
        code += "# ============================================\n"
        code += f"best_name = {best_name!r}\n"

        model_instantiation = _get_model_instantiation(best_name, best_params, problem_type)
        code += model_instantiation + "\n\n"

        # Cross-validation
        if is_classification:
            code += "# Cross-validation (same as pipeline)\n"
            code += "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            code += "cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')\n"
            code += "print(f'Cross-validation scores: {cv_scores}')\n"
            code += "print(f'CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')\n\n"
        else:
            code += "# Cross-validation (same as pipeline)\n"
            code += "cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')\n"
            code += "print(f'Cross-validation scores: {cv_scores}')\n"
            code += "print(f'CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')\n\n"

        code += "# Train and predict\n"
        code += "model.fit(X_train, y_train)\n"
        code += "y_pred = model.predict(X_test)\n"

        if target_transform:
            code += "\n# Inverse transform predictions\n"
            code += "y_pred = pt.inverse_transform(y_pred.reshape(-1, 1)).ravel()\n"

        # --- Evaluation ---
        code += "\n# ============================================\n"
        code += "# 8. EVALUATION\n"
        code += "# ============================================\n"

        if is_classification:
            code += "acc = accuracy_score(y_test, y_pred)\n"
            code += "print(f'\\nBest Model ({best_name}) Accuracy: {acc:.4f}')\n\n"

            code += "# Detailed metrics\n"
            code += "avg = 'binary' if len(y.unique()) == 2 else 'weighted'\n"
            code += "prec = precision_score(y_test, y_pred, average=avg, zero_division=0)\n"
            code += "rec = recall_score(y_test, y_pred, average=avg, zero_division=0)\n"
            code += "f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)\n"
            code += "print(f'Precision: {prec:.4f}')\n"
            code += "print(f'Recall:    {rec:.4f}')\n"
            code += "print(f'F1 Score:  {f1:.4f}')\n\n"

            code += "# Classification Report\n"
            code += "report = classification_report(y_test, y_pred)\n"
            code += "print('\\nClassification Report:\\n', report)\n\n"

            code += "# Save metrics\n"
            code += "with open('stats/model_performance.txt', 'w') as f:\n"
            code += f"    f.write(f'Best Model ({best_name}): {{acc:.4f}}\\n')\n"
            code += "    f.write(f'Precision: {prec:.4f}\\n')\n"
            code += "    f.write(f'Recall: {rec:.4f}\\n')\n"
            code += "    f.write(f'F1 Score: {f1:.4f}\\n')\n"
            code += "    f.write(f'CV Mean: {cv_scores.mean():.4f}\\n')\n"
            code += "with open('reports/metrics.txt', 'w') as f:\n"
            code += "    f.write('Classification Report\\n')\n"
            code += "    f.write('=' * 50 + '\\n\\n')\n"
            code += "    f.write(report)\n"
        else:
            code += "r2 = r2_score(y_test, y_pred)\n"
            code += "mse = mean_squared_error(y_test, y_pred)\n"
            code += "rmse = mse ** 0.5\n"
            code += "mae = mean_absolute_error(y_test, y_pred)\n"
            code += "print(f'\\nBest Model ({best_name})')\n"
            code += "print(f'R² Score: {r2:.4f}')\n"
            code += "print(f'RMSE:     {rmse:.4f}')\n"
            code += "print(f'MAE:      {mae:.4f}')\n\n"

            code += "# Save metrics\n"
            code += "with open('stats/model_performance.txt', 'w') as f:\n"
            code += f"    f.write(f'Best Model ({best_name})\\n')\n"
            code += "    f.write(f'R2 Score: {r2:.4f}\\n')\n"
            code += "    f.write(f'MSE: {mse:.4f}\\n')\n"
            code += "    f.write(f'RMSE: {rmse:.4f}\\n')\n"
            code += "    f.write(f'MAE: {mae:.4f}\\n')\n"
            code += "    f.write(f'CV Mean: {cv_scores.mean():.4f}\\n')\n"

        # --- Plots ---
        code += "\n# ============================================\n"
        code += "# 9. VISUALIZATION\n"
        code += "# ============================================\n"
        code += "try:\n"
        code += "    # Correlation heatmap\n"
        code += "    num_data = data.select_dtypes(include='number')\n"
        code += "    plt.figure(figsize=(10, 8))\n"
        code += "    sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')\n"
        code += "    plt.title('Correlation Matrix')\n"
        code += "    plt.tight_layout()\n"
        code += "    plt.savefig('plots/correlation_heatmap.png', dpi=150)\n"
        code += "    plt.close()\n"

        if is_classification:
            code += "\n    # Confusion Matrix\n"
            code += "    cm = confusion_matrix(y_test, y_pred)\n"
            code += "    plt.figure(figsize=(8, 6))\n"
            code += "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n"
            code += "    plt.title('Confusion Matrix')\n"
            code += "    plt.ylabel('Actual')\n"
            code += "    plt.xlabel('Predicted')\n"
            code += "    plt.tight_layout()\n"
            code += "    plt.savefig('plots/confusion_matrix.png', dpi=150)\n"
            code += "    plt.close()\n"
        else:
            code += "\n    # Actual vs Predicted\n"
            code += "    plt.figure(figsize=(8, 6))\n"
            code += "    plt.scatter(y_test, y_pred, alpha=0.5)\n"
            code += "    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\n"
            code += "    plt.xlabel('Actual')\n"
            code += "    plt.ylabel('Predicted')\n"
            code += "    plt.title('Actual vs Predicted')\n"
            code += "    plt.tight_layout()\n"
            code += "    plt.savefig('plots/actual_vs_predicted.png', dpi=150)\n"
            code += "    plt.close()\n"

            code += "\n    # Residual Plot\n"
            code += "    residuals = y_test - y_pred\n"
            code += "    plt.figure(figsize=(8, 6))\n"
            code += "    plt.scatter(y_pred, residuals, alpha=0.5)\n"
            code += "    plt.axhline(y=0, color='r', linestyle='--')\n"
            code += "    plt.xlabel('Predicted')\n"
            code += "    plt.ylabel('Residuals')\n"
            code += "    plt.title('Residual Plot')\n"
            code += "    plt.tight_layout()\n"
            code += "    plt.savefig('plots/residual_plot.png', dpi=150)\n"
            code += "    plt.close()\n"

        code += "\n    print('\\nPlots saved to plots/')\n"
        code += "except Exception as e:\n"
        code += "    print(f'Plotting error: {e}')\n"

        code += "\nprint('\\n✓ Analysis complete!')\n"

        # --- README ---
        # Build a detailed summary of what was done
        preproc_details = []
        if null_strategy:
            methods_used = set()
            for _, details in null_strategy.items():
                m = details.get('method', 'median') if isinstance(details, dict) else details
                methods_used.add(m)
            preproc_details.append(f"- Null Handling: {', '.join(methods_used)}")
        if outlier_cols:
            preproc_details.append(f"- Outlier Capping: IQR (threshold={outlier_threshold}) on {len(outlier_cols)} columns")
        if features_to_drop:
            preproc_details.append(f"- Dropped Features: {features_to_drop}")

        encoding_details = []
        if onehot_cols:
            encoding_details.append(f"- One-Hot: {onehot_cols}")
        if label_cols:
            encoding_details.append(f"- Label Encoding: {label_cols}")
        if target_enc_cols:
            encoding_details.append(f"- Frequency Encoding: {target_enc_cols}")

        readme_content = f"""# AutoEDA Analysis Results

## Best Model: {best_name}
## Score: {best_model_info.get('score', 0):.4f}
## Problem Type: {problem_type}

### How to Run
```bash
pip install -r requirements.txt
python analysis.py
```

### Pipeline Decisions (Reproduced in analysis.py)

#### Preprocessing
{chr(10).join(preproc_details) if preproc_details else '- No special preprocessing required'}

#### Feature Encoding
{chr(10).join(encoding_details) if encoding_details else '- Auto label encoding for categorical columns'}

#### Advanced Processing
- Train/Test Split: 80/20 {'(Stratified)' if is_classification else ''}
- Scaling: {'StandardScaler' if used_scaling else 'None'}
- Class Balancing: {'SMOTE (applied only when ratio < 0.5)' if used_balancing else 'None'}
- Target Transform: {'Yeo-Johnson PowerTransformer' if target_transform else 'None'}

#### Model
- **{best_name}**
- Parameters: `{best_params}`

### Output Files
- `stats/model_performance.txt` — Score summary
- `reports/metrics.txt` — Detailed classification/regression report
- `plots/correlation_heatmap.png` — Feature correlations
{'- `plots/confusion_matrix.png` — Confusion matrix' if is_classification else '- `plots/actual_vs_predicted.png` — Actual vs Predicted scatter'}
{'- `plots/residual_plot.png` — Residual analysis' if not is_classification else ''}
"""

        log(f"[PROJECT] ✓ Generated code for {best_name} (mirrors pipeline exactly)")

        return A2AResponse(
            task_id=task.task_id,
            sender="project-agent",
            status="COMPLETED",
            output={
                "analysis_code": code,
                "readme": readme_content
            }
        )
    except Exception as e:
        log(f"[PROJECT] Error: {e}")
        print(f"Error: {e}")
        raise


def _get_model_import(best_name, problem_type):
    """Return the correct import statement for the best model."""
    imports = []

    if best_name == 'ensemble':
        if problem_type == 'classification':
            imports.append("from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier")
            imports.append("from sklearn.svm import SVC")
            imports.append("from xgboost import XGBClassifier")
        else:
            imports.append("from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor")
            imports.append("from xgboost import XGBRegressor")
    elif 'random_forest' in best_name:
        cls = "RandomForestClassifier" if problem_type == 'classification' else "RandomForestRegressor"
        imports.append(f"from sklearn.ensemble import {cls}")
    elif 'gradient_boosting' in best_name:
        cls = "GradientBoostingClassifier" if problem_type == 'classification' else "GradientBoostingRegressor"
        imports.append(f"from sklearn.ensemble import {cls}")
    elif 'xgboost' in best_name:
        imports.append("from xgboost import XGBClassifier, XGBRegressor")
    elif 'logistic' in best_name:
        imports.append("from sklearn.linear_model import LogisticRegression")
    elif 'linear_regression' in best_name:
        imports.append("from sklearn.linear_model import LinearRegression")
    elif 'ridge' in best_name:
        imports.append("from sklearn.linear_model import Ridge")
    elif 'lasso' in best_name:
        imports.append("from sklearn.linear_model import Lasso")
    elif 'elastic' in best_name:
        imports.append("from sklearn.linear_model import ElasticNet")
    elif 'decision_tree' in best_name:
        cls = "DecisionTreeClassifier" if problem_type == 'classification' else "DecisionTreeRegressor"
        imports.append(f"from sklearn.tree import {cls}")
    elif 'naive_bayes' in best_name:
        imports.append("from sklearn.naive_bayes import GaussianNB")
    elif 'svm' in best_name or 'svc' in best_name:
        imports.append("from sklearn.svm import SVC")
    elif 'knn' in best_name:
        imports.append("from sklearn.neighbors import KNeighborsClassifier")
    else:
        # Fallback: import all common models
        imports.append("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor")
        imports.append("from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor")
        imports.append("from xgboost import XGBClassifier, XGBRegressor")

    return "\n".join(imports)


def _get_model_instantiation(best_name, best_params, problem_type):
    """Return the correct model instantiation code with exact parameters."""
    params = dict(best_params)
    param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())

    if best_name == 'ensemble':
        if problem_type == 'classification':
            voting = params.get('ensemble_voting', 'soft')
            return (
                "model = VotingClassifier(\n"
                "    estimators=[\n"
                "        ('gradient_boosting', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)),\n"
                "        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),\n"
                "        ('xgboost', XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, eval_metric='logloss')),\n"
                "    ],\n"
                f"    voting={repr(voting)}\n"
                ")"
            )
        return (
            "model = VotingRegressor(\n"
            "    estimators=[\n"
            "        ('random_forest', RandomForestRegressor(n_estimators=200, random_state=42)),\n"
            "        ('gradient_boosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),\n"
            "        ('xgboost', XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),\n"
            "    ]\n"
            ")"
        )
    elif 'random_forest' in best_name:
        cls = "RandomForestClassifier" if problem_type == 'classification' else "RandomForestRegressor"
        return f"model = {cls}({param_str})"
    elif 'gradient_boosting' in best_name:
        cls = "GradientBoostingClassifier" if problem_type == 'classification' else "GradientBoostingRegressor"
        return f"model = {cls}({param_str})"
    elif 'xgboost' in best_name:
        cls = "XGBClassifier" if problem_type == 'classification' else "XGBRegressor"
        extra = ", eval_metric='logloss'" if problem_type == 'classification' else ""
        return f"model = {cls}({param_str}{extra})"
    elif 'logistic' in best_name:
        if 'max_iter' not in params:
            param_str += ", max_iter=1000" if param_str else "max_iter=1000"
        return f"model = LogisticRegression({param_str})"
    elif 'linear_regression' in best_name:
        return "model = LinearRegression()"
    elif 'ridge' in best_name:
        return f"model = Ridge({param_str})"
    elif 'lasso' in best_name:
        if 'max_iter' not in params:
            param_str += ", max_iter=2000" if param_str else "max_iter=2000"
        return f"model = Lasso({param_str})"
    elif 'elastic' in best_name:
        if 'max_iter' not in params:
            param_str += ", max_iter=2000" if param_str else "max_iter=2000"
        return f"model = ElasticNet({param_str})"
    elif 'decision_tree' in best_name:
        cls = "DecisionTreeClassifier" if problem_type == 'classification' else "DecisionTreeRegressor"
        return f"model = {cls}({param_str})"
    elif 'naive_bayes' in best_name:
        return "model = GaussianNB()"
    elif 'svm' in best_name or 'svc' in best_name:
        if 'probability' not in params and problem_type == 'classification':
            param_str += ", probability=True" if param_str else "probability=True"
        return f"model = SVC({param_str})"
    elif 'knn' in best_name:
        return f"model = KNeighborsClassifier({param_str})"
    else:
        cls = "RandomForestClassifier" if problem_type == 'classification' else "RandomForestRegressor"
        return f"model = {cls}(n_estimators=200, random_state=42)"
