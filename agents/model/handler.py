import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               StackingClassifier, StackingRegressor,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier

# LightGBM (optional)
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, mean_absolute_error,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer
from a2a.schemas import A2ATask, A2AResponse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from core.data_utils import load_csv_robust, normalize_missing_markers
from agents.preprocessing.handler import (
    fit_null_handling, transform_null_handling,
    fit_outlier_handling, transform_outlier_handling
)
from agents.feature.handler import (
    fit_datetime, transform_datetime,
    fit_text_vectorization, transform_text_vectorization,
    fit_drop_features, transform_drop_features,
    fit_encoding, transform_encoding
)

# Smart ML imports
try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[MODEL] Warning: optuna not installed. Hyperparameter tuning disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[MODEL] Warning: shap not installed. SHAP analysis disabled.")





# =============================================================================
# PREPROCESSING — fits on training data only, transforms both train & test
# =============================================================================

def preprocess_data(df, prep_strategy, feat_strategy, target_col, log_callback=None):
    """
    Smart preprocessing using LLM strategies from preprocessing and feature agents.

    IMPORTANT: This function only extracts X/y and encodes the target.
    Column-level preprocessing (nulls, outliers, encoding) is done AFTER
    the train/test split via fit_preprocess / transform_preprocess to
    prevent data leakage.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    # Defensive normalization in case this helper is called standalone.
    df = normalize_missing_markers(df)
    log(f"[MODEL] Separating features and target...")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    log(f"[MODEL] Shape: X={X.shape}, y={y.shape}")
    return X, y


def fit_preprocess(X_train, y_train, prep_strategy, feat_strategy, target_col, log_callback=None):
    """
    Fit preprocessing on TRAINING data only. Returns transformed X_train and
    a state dict that can be used to transform the test set identically.
    
    This prevents data leakage: statistics (mean, median, frequencies, IQR
    bounds) are computed from training data only.
    
    Logic is delegated to preprocessing and feature agent fit functions.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    log(f"[MODEL] Fitting preprocessing on training data...")

    X = normalize_missing_markers(X_train.copy())
    original_columns = X.columns.tolist()

    # Parse strategies
    prep = prep_strategy if isinstance(prep_strategy, dict) else {}
    feat = feat_strategy if isinstance(feat_strategy, dict) else {}

    # --- STEP 0: NULL HANDLING (fit on train) ---
    X, y_train, null_state = fit_null_handling(X, y_train, prep, target_col, log_callback)

    # --- STEP 1: DROP FEATURES ---
    X, drop_state = fit_drop_features(X, feat, target_col, log_callback)

    # --- STEP 2: OUTLIER HANDLING (bounds from train) ---
    X, outlier_state = fit_outlier_handling(X, prep, target_col, log_callback)

    # --- STEP 3: DATETIME EXTRACTION ---
    X, datetime_state = fit_datetime(X, feat, target_col, log_callback)

    # --- STEP 4: TEXT VECTORIZATION (TF-IDF) ---
    X, text_state = fit_text_vectorization(X, feat, target_col, log_callback)

    # --- STEP 5: ENCODING (fit on train) ---
    X, encoding_state = fit_encoding(X, feat, target_col, log_callback)

    # Fill any remaining NaN with median (fitted on train)
    remaining_medians = {}
    if X.isna().sum().sum() > 0:
        remaining_medians = X.median(numeric_only=True).to_dict()
        X = X.fillna(remaining_medians)
        log(f"[MODEL] Filled remaining NaN with train median")

    # Final guard: never return an empty feature matrix
    if X.shape[1] == 0:
        log("[MODEL] WARNING: Preprocessing resulted in 0 features. Restoring original.")
        X = X_train[original_columns].copy()
        null_state = {"fill_values": {}}
        drop_state = {"dropped_cols": []}
        outlier_state = {"clip_bounds": {}}
        datetime_state = {"datetime_cols": []}
        text_state = {"text_cols": {}}
        encoding_state = {"onehot_cols": [], "label_maps": {}, "freq_maps": {}}
        remaining_medians = {}

    # Combine all state into one dict
    state = {
        **null_state,
        **drop_state,
        **outlier_state,
        **datetime_state,
        **text_state,
        **encoding_state,
        "remaining_medians": remaining_medians,
        "columns": X.columns.tolist()
    }

    return X, y_train, state

def transform_preprocess(X_test, state, log_callback=None):
    """
    Apply the same preprocessing to test data using statistics from training.
    No fitting — only transforming using 'state' from fit_preprocess.
    
    Logic is delegated to preprocessing and feature agent transform functions.
    """
    X = normalize_missing_markers(X_test.copy())

    # Apply all transforms using train-fitted state
    X = transform_null_handling(X, state, log_callback)
    X = transform_drop_features(X, state, log_callback)
    X = transform_outlier_handling(X, state, log_callback)
    X = transform_datetime(X, state, log_callback)
    X = transform_text_vectorization(X, state, log_callback)
    X = transform_encoding(X, state, log_callback)

    # Fill remaining NaN
    remaining = state.get("remaining_medians", {})
    if remaining:
        X = X.fillna(remaining)
    if X.isna().sum().sum() > 0:
        X = X.fillna(0)

    # Align columns to match training
    train_cols = state["columns"]
    for col in train_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[train_cols]  # Reorder and drop extras

    return X

# =============================================================================
# TUNING
# =============================================================================

def hyperparameter_tuning(X_train, y_train, problem_type, log_callback=None):
    """
    Optuna-based hyperparameter optimization using CROSS-VALIDATION
    (not test set) to prevent overfitting the tuning to held-out data.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    if not OPTUNA_AVAILABLE:
        return None

    log(f"[MODEL] Starting Optuna hyperparameter tuning (30 trials)...")

    # Suppress Optuna's default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        choices = ['random_forest', 'xgboost', 'gradient_boosting']
        if LGBM_AVAILABLE:
            choices.append('lightgbm')
        model_type = trial.suggest_categorical('model', choices)

        if model_type == 'random_forest':
            # Dynamic min samples based on dataset size to prevent overfitting tiny datasets
            n_samples = len(X_train)
            max_leaf_bound = max(4, int(n_samples * 0.05)) # up to 5% of data in leaf
            
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, max_leaf_bound),
                'random_state': 42
            }
            if problem_type == 'classification':
                params['class_weight'] = 'balanced'
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)

        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            if problem_type == 'classification':
                # Dynamically calculate scale_pos_weight for binary classification
                if len(np.unique(y_train)) == 2:
                    counts = np.bincount(y_train)
                    scale_weight = float(counts[0]) / counts[1] if counts[1] > 0 else 1.0
                    params['scale_pos_weight'] = scale_weight
                model = XGBClassifier(**params, eval_metric='logloss')
            else:
                model = XGBRegressor(**params)

        elif model_type == 'gradient_boosting':
            params = {
                'n_estimators': trial.suggest_int('gb_n_estimators', 100, 400),
                'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('gb_max_depth', 3, 8),
                'subsample': trial.suggest_float('gb_subsample', 0.6, 1.0),
                'random_state': 42
            }
            if problem_type == 'classification':
                model = GradientBoostingClassifier(**params)
            else:
                model = GradientBoostingRegressor(**params)

        elif model_type == 'lightgbm' and LGBM_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('lgbm_max_depth', 3, 10),
                'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            if problem_type == 'classification':
                params['class_weight'] = 'balanced'
                model = LGBMClassifier(**params)
            else:
                model = LGBMRegressor(**params)
        else:
            params = {'random_state': 42}
            if problem_type == 'classification':
                params['class_weight'] = 'balanced'
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)

        # Use CROSS-VALIDATION for scoring (prevents test-set overfitting)
        scoring = 'accuracy' if problem_type == 'classification' else 'r2'
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if problem_type == 'classification' else 5
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return cv_scores.mean()

    # Run optimization
    study = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study.optimize(objective, n_trials=30, timeout=120, show_progress_bar=False)

    log(f"[MODEL] ✓ Optuna best CV score: {study.best_value:.4f}")
    log(f"[MODEL] Best params: {study.best_params}")

    # Train best model on full training data
    best_params = study.best_params.copy()
    model_type = best_params.pop('model')

    # Clean param names (remove prefix)
    clean_params = {}
    for k, v in best_params.items():
        # Remove prefix like 'rf_', 'xgb_', 'gb_'
        clean_key = k.split('_', 1)[1] if '_' in k and k.split('_')[0] in ('rf', 'xgb', 'gb', 'lgbm') else k
        clean_params[clean_key] = v
    clean_params['random_state'] = 42

    if model_type == 'random_forest':
        if problem_type == 'classification':
            clean_params.setdefault('class_weight', 'balanced')
            best_model = RandomForestClassifier(**clean_params)
        else:
            best_model = RandomForestRegressor(**clean_params)
    elif model_type == 'xgboost':
        if problem_type == 'classification':
            if len(np.unique(y_train)) == 2:
                 counts = np.bincount(y_train)
                 scale_weight = float(counts[0]) / counts[1] if counts[1] > 0 else 1.0
                 clean_params.setdefault('scale_pos_weight', scale_weight)
            best_model = XGBClassifier(**clean_params, eval_metric='logloss')
        else:
            best_model = XGBRegressor(**clean_params)
    elif model_type == 'lightgbm' and LGBM_AVAILABLE:
        clean_params['verbose'] = -1
        if problem_type == 'classification':
            clean_params.setdefault('class_weight', 'balanced')
            best_model = LGBMClassifier(**clean_params)
        else:
            best_model = LGBMRegressor(**clean_params)
    else:
        if problem_type == 'classification':
            best_model = GradientBoostingClassifier(**clean_params)
        else:
            best_model = GradientBoostingRegressor(**clean_params)

    best_model.fit(X_train, y_train)

    return {
        'model': best_model,
        'cv_score': study.best_value,
        'params': clean_params,
        'name': f'{model_type}_tuned'
    }


def create_ensemble(trained_models, X_train, y_train, problem_type, log_callback=None):
    """
    Create voting ensemble from TOP trained model OBJECTS (not re-created).
    Returns (ensemble_model_or_None, ensemble_members_metadata).
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    log(f"[MODEL] Creating ensemble from top trained models...")

    # Sort by score, pick top 3
    sorted_models = sorted(trained_models.items(), key=lambda x: x[1]["score"], reverse=True)

    linear_model_names = {'logistic_regression', 'linear_regression', 'naive_bayes'}
    estimators = []
    for name, info in sorted_models[:3]:
        model_obj = info.get("model_obj")
        if model_obj is None:
            continue
        # Skip simple linear models (they don't add much to ensembles)
        if name in linear_model_names or any(tok in name for tok in ("logistic", "linear", "naive_bayes")):
            continue
        estimators.append((name, model_obj))

    if len(estimators) < 2:
        log(f"[MODEL] Not enough models for ensemble (need >= 2, got {len(estimators)})")
        return None, []

    if problem_type == 'classification':
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)
        ensemble = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
    else:
        final_estimator = Ridge(alpha=1.0, random_state=42)
        ensemble = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)

    ensemble.fit(X_train, y_train)
    log(f"[MODEL] Stacking Ensemble created from {len(estimators)} models: {[n for n, _ in estimators]}")

    ensemble_members = []
    for name, model_obj in estimators:
        params = model_obj.get_params(deep=False) if hasattr(model_obj, "get_params") else {}
        safe_params = {}
        for k, v in params.items():
            if isinstance(v, (np.integer,)):
                safe_params[k] = int(v)
            elif isinstance(v, (np.floating,)):
                safe_params[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                safe_params[k] = bool(v)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                safe_params[k] = v
            else:
                safe_params[k] = str(v)
        ensemble_members.append({
            "name": name,
            "class_name": model_obj.__class__.__name__,
            "params": safe_params
        })

    return ensemble, ensemble_members


def analyze_feature_importance(best_model, X_test_scaled, feature_names, output_folder=None, log_callback=None):
    """Analyze and visualize feature importance using SHAP (on scaled data matching training)."""
    def log(msg):
        if log_callback:
            log_callback(msg)

    if not SHAP_AVAILABLE or not hasattr(shap, 'TreeExplainer'):
        log("[MODEL] ⚠ SHAP not available, skipping feature importance analysis")
        return None

    try:
        log(f"[MODEL] Analyzing feature importance with SHAP...")

        import os
        if output_folder is None:
            output_folder = 'plots'
        os.makedirs(output_folder, exist_ok=True)

        # Check if model is tree-based
        tree_models = [RandomForestClassifier, RandomForestRegressor,
                      XGBClassifier, XGBRegressor,
                      GradientBoostingClassifier, GradientBoostingRegressor,
                      DecisionTreeClassifier, DecisionTreeRegressor]
        if LGBM_AVAILABLE:
            tree_models.extend([LGBMClassifier, LGBMRegressor])
        tree_models = tuple(tree_models)

        if not isinstance(best_model, tree_models):
            log("[MODEL] ⚠ Model is not tree-based, skipping SHAP analysis")
            return None

        # Normalize SHAP input to numeric matrix to avoid object/string conversion errors.
        if isinstance(X_test_scaled, pd.DataFrame):
            X_for_shap = X_test_scaled.copy()
        else:
            X_for_shap = pd.DataFrame(X_test_scaled, columns=feature_names[: np.asarray(X_test_scaled).shape[1]])

        for col in X_for_shap.columns:
            if not pd.api.types.is_numeric_dtype(X_for_shap[col]):
                # Attempt to clean brackets and coerce
                # For things like "[5.054E-1]" or "['5.054']"
                cleaned = X_for_shap[col].astype(str).str.strip().str.replace(r"[\[\]'\"]", "", regex=True)
                X_for_shap[col] = pd.to_numeric(cleaned, errors='coerce')

        if X_for_shap.isna().sum().sum() > 0:
            X_for_shap = X_for_shap.fillna(0)

        # Force all to float, and explicitly handle any remaining non-convertible types
        try:
             X_for_shap = X_for_shap.astype(float)
        except Exception as e:
             log(f"[MODEL] ⚠ SHAP cast failed ({e}). Forcing numeric coercion.")
             # Bruteforce matrix conversion if .astype(float) still sees mixed types
             X_for_shap = X_for_shap.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_for_shap)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Check for 3D array (samples, features, classes) -> collapse classes
        if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            shap_values = np.abs(shap_values).sum(axis=-1)

        # Calculate feature importance
        mean_abs_importance = np.abs(shap_values).mean(axis=0)

        # Verify dimensions match
        if len(mean_abs_importance) != len(feature_names):
             log(f"[MODEL] ⚠ SHAP dimension mismatch ({len(mean_abs_importance)} vs {len(feature_names)}). Skipping.")
             return None

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_importance
        }).sort_values('importance', ascending=False)

        log(f"[MODEL] Top 10 Important Features:")
        for _, row in importance_df.head(10).iterrows():
            log(f"[MODEL]   {row['feature']}: {row['importance']:.4f}")

        # Save summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(f'{output_folder}/shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save bar plot
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP value| (average impact on model output)')
        plt.title('Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_folder}/shap_importance_bar.png', dpi=150, bbox_inches='tight')
        plt.close()

        log(f"[MODEL] ✓ SHAP plots saved to {output_folder}/")

        return importance_df.to_dict('records')

    except Exception as e:
        log(f"[MODEL] ⚠ SHAP analysis failed: {e}")
        return None


# =============================================================================
# PROBLEM TYPE DETECTION
# =============================================================================

def detect_problem_type(y, target_col, log_callback=None):
    """
    Robust problem type detection.

    Rules:
    - If target was originally string/category → classification
    - If float with non-integer values → regression
    - If integer with ≤ 20 unique values AND unique ratio < 5% of dataset → classification
    - Otherwise → regression
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    n_unique = len(y.unique())
    unique_ratio = n_unique / len(y) if len(y) > 0 else 0

    # If target is object or has very few unique values, it's classification
    if y.dtype == 'object' or str(y.dtype) == 'category':
        log(f"[MODEL] Target '{target_col}' is categorical → classification")
        return "classification"

    # Float with fractional values → regression
    if y.dtype in ['float64', 'float32']:
        has_fractions = (y != y.astype(int)).any()
        if has_fractions or n_unique > 20:
            log(f"[MODEL] Target '{target_col}' is float with {n_unique} unique values → regression")
            return "regression"

    # Integer-like: use heuristics
    if n_unique <= 2:
        log(f"[MODEL] Target '{target_col}' is binary ({n_unique} classes) → classification")
        return "classification"

    if n_unique <= 20 and unique_ratio < 0.05:
        log(f"[MODEL] Target '{target_col}' has {n_unique} classes (ratio={unique_ratio:.3f}) → classification")
        return "classification"

    log(f"[MODEL] Target '{target_col}' has {n_unique} unique values → regression")
    return "regression"


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle(task: A2ATask, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)

    try:
        log(f"[MODEL] Starting SMART model training...")
        csv_path = task.input["csv_path"]
        prep_strategy = task.input.get("prep_strategy", {})
        feat_strategy = task.input.get("feat_strategy", {})
        output_folder = task.input.get("output_folder", "plots")

        log(f"[MODEL] Loading dataset...")
        df = load_csv_robust(csv_path)
        df = normalize_missing_markers(df)

        # Use provided target column or fall back to last column
        target_col = task.input.get("target_column", df.columns[-1])
        log(f"[MODEL] Target column: {target_col}")

        # --- Separate X/y ---
        X, y = preprocess_data(df, prep_strategy, feat_strategy, target_col, log_callback)

        # --- Detect problem type ROBUSTLY ---
        problem_type = detect_problem_type(y, target_col, log_callback)
        is_classification = problem_type == "classification"

        # --- Encode target ALWAYS for classification (XGBoost expects 0, 1, 2...) ---
        target_le = None
        if is_classification:
            target_le = LabelEncoder()
            y = pd.Series(target_le.fit_transform(y), index=y.index)
            log(f"[MODEL] Target encoded: {dict(zip(target_le.classes_, range(len(target_le.classes_))))}")

        # --- SPLIT FIRST to prevent data leakage ---
        split_kwargs = {"test_size": 0.2, "random_state": 42}
        if is_classification:
            split_kwargs["stratify"] = y  # Stratified split for classification
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        log(f"[MODEL] Train/test split: {len(X_train)}/{len(X_test)}" +
            (" (stratified)" if is_classification else ""))

        # --- THEN preprocess (fit on train, transform test) ---
        X_train, y_train, preprocess_state = fit_preprocess(
            X_train, y_train, prep_strategy, feat_strategy, target_col, log_callback
        )
        X_test = transform_preprocess(X_test, preprocess_state, log_callback)
        log(f"[MODEL] After preprocessing: train={X_train.shape}, test={X_test.shape}")

        # --- Feature Scaling ---
        # Robust scaling chaining into Standard scaling
        robust_scaler = RobustScaler()
        std_scaler = StandardScaler()
        
        # Fit-transform train
        X_train_r = robust_scaler.fit_transform(X_train)
        X_train_s = std_scaler.fit_transform(X_train_r)
        X_train_scaled = pd.DataFrame(X_train_s, columns=X_train.columns)
        
        # Transform test
        X_test_r = robust_scaler.transform(X_test)
        X_test_s = std_scaler.transform(X_test_r)
        X_test_scaled = pd.DataFrame(X_test_s, columns=X_test.columns)
        
        log(f"[MODEL] \u2713 Applied RobustScaler + StandardScaler pipeline")

        # --- Class imbalance handling policy ---
        # Calculate dynamic scale_pos_weight for XGBoost
        scale_pos_weight = 1.0
        actually_balanced = False
        if is_classification:
            _vc = y_train.value_counts()
            if len(_vc) >= 2:
                # Assuming binary classification for simple dynamic weighting
                if len(_vc) == 2:
                     scale_pos_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
                _ratio = _vc.min() / _vc.max()
                log(f"[MODEL] Class ratio: {_ratio:.2f} (SMOTE disabled; using class-weighted models with scale_pos_weight={scale_pos_weight:.2f})")

        # --- Target Transformation for skewed regression targets ---
        target_transformer = None
        y_train_use = y_train.copy()
        if not is_classification:
            skewness = float(pd.Series(y_train).skew())
            if abs(skewness) > 1.0:
                log(f"[MODEL] ⚠ Target skewness: {skewness:.2f} — applying PowerTransformer")
                target_transformer = PowerTransformer(method='yeo-johnson')
                y_train_use = target_transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                log(f"[MODEL] ✓ Applied Yeo-Johnson transformation to target")

        # --- Define base models ---
        if is_classification:
            models = {
                "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
                "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
                "svm": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
                "knn": KNeighborsClassifier(n_neighbors=5),
                "naive_bayes": GaussianNB(),
                "decision_tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                "xgboost": XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
            }
            if LGBM_AVAILABLE:
                models["lightgbm"] = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    random_state=42,
                    class_weight='balanced',
                    verbose=-1
                )
            metric_name = "accuracy"
        else:
            models = {
                "linear_regression": LinearRegression(),
                "ridge": Ridge(alpha=1.0, random_state=42),
                "lasso": Lasso(alpha=0.1, max_iter=2000, random_state=42),
                "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42),
                "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
                "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
                "decision_tree": DecisionTreeRegressor(random_state=42),
                "xgboost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
            }
            if LGBM_AVAILABLE:
                models["lightgbm"] = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1)
            metric_name = "r2_score"

        log(f"[MODEL] Training {len(models)} base models with cross-validation...")

        # --- Train base models ---
        results = {}
        best_cv_score = -np.inf
        best_model_name = None
        best_model_obj = None

        # Use scaled features
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled

        # CV strategy
        if is_classification:
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv_strategy = 5

        for i, (name, model) in enumerate(models.items(), 1):
            log(f"[MODEL] [{i}/{len(models)}] Training {name}...")

            try:
                # Cross-validation on training data
                scoring = 'accuracy' if is_classification else 'r2'
                cv_scores = cross_val_score(model, X_train_use, y_train_use, cv=cv_strategy, scoring=scoring)
                cv_mean = cv_scores.mean()

                # Train on full training set
                model.fit(X_train_use, y_train_use)
                
                # Train score for overfitting diagnostics
                y_pred_train = model.predict(X_train_use)
                y_pred = model.predict(X_test_use)

                # Inverse transform predictions for regression
                if target_transformer is not None and not is_classification:
                    y_pred_train = target_transformer.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
                    y_pred = target_transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()

                if is_classification:
                    test_score = accuracy_score(y_test, y_pred)
                    avg = 'binary' if len(y.unique()) == 2 else 'weighted'
                    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
                    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
                    auc = None
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test_use)
                            if len(y.unique()) == 2:
                                auc = roc_auc_score(y_test, y_proba[:, 1])
                            else:
                                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    except Exception:
                        pass
                    auc_str = f", AUC={auc:.4f}" if auc is not None else ""
                    train_acc = accuracy_score(y_train_use, y_pred_train)
                    overfit_gap = train_acc - test_score
                    overfit_warning = " \u26a0 OVERFIT" if overfit_gap > 0.1 else ""
                    log(f"[MODEL]   {name}: Acc={test_score:.4f}, F1={f1:.4f}{auc_str}, CV={cv_mean:.4f}, train={train_acc:.4f}{overfit_warning}")
                    results[name] = {
                        "score": float(test_score),
                        "accuracy": float(test_score),
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1_score": float(f1),
                        "auc_roc": float(auc) if auc is not None else None,
                        "cv_score": float(cv_mean),
                        "metric": metric_name,
                        "model_obj": model  # Keep reference for ensemble
                    }
                else:
                    test_score = r2_score(y_test, y_pred)
                    train_r2 = r2_score(y_train, y_pred_train)
                    overfit_gap = train_r2 - test_score
                    overfit_warning = " OVERFIT" if overfit_gap > 0.1 else ""
                    n = len(y_test)
                    p = X_test_use.shape[1]
                    adj_r2 = 1 - (1 - test_score) * (n - 1) / (n - p - 1) if n > p + 1 else test_score
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = mse ** 0.5
                    mae = mean_absolute_error(y_test, y_pred)
                    log(f"[MODEL]   {name}: R2={test_score:.4f}, RMSE={rmse:.4f}, CV={cv_mean:.4f}, train={train_r2:.4f}{overfit_warning}")
                    results[name] = {
                        "score": float(test_score),
                        "r2": float(test_score),
                        "train_r2": float(train_r2),
                        "adjusted_r2": float(adj_r2),
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "cv_score": float(cv_mean),
                        "metric": metric_name,
                        "model_obj": model
                    }

                # Use CV score for model SELECTION (more reliable than single test split)
                if cv_mean > best_cv_score:
                    best_cv_score = cv_mean
                    best_model_name = name
                    best_model_obj = model
            except Exception as e:
                log(f"[MODEL]   {name}: FAILED — {e}")
                results[name] = {"score": 0.0, "cv_score": 0.0, "metric": metric_name, "error": str(e)}

        log(f"[MODEL] Best base model: {best_model_name} (CV={best_cv_score:.4f})")

        # --- Always run Optuna tuning (not gated behind 80%) ---
        tuned_result = None
        if OPTUNA_AVAILABLE:
            log(f"[MODEL] Running Optuna hyperparameter tuning...")
            tuned_result = hyperparameter_tuning(X_train_use, y_train_use, problem_type, log_callback)

            if tuned_result:
                # Evaluate tuned model on test set
                y_pred_tuned = tuned_result['model'].predict(X_test_use)
                if target_transformer is not None and not is_classification:
                    y_pred_tuned = target_transformer.inverse_transform(y_pred_tuned.reshape(-1, 1)).ravel()

                if is_classification:
                    tuned_test_score = accuracy_score(y_test, y_pred_tuned)
                else:
                    tuned_test_score = r2_score(y_test, y_pred_tuned)

                tuned_cv = tuned_result['cv_score']
                log(f"[MODEL] Tuned model: test={tuned_test_score:.4f}, CV={tuned_cv:.4f}")

                results[tuned_result['name']] = {
                    "score": float(tuned_test_score),
                    "cv_score": float(tuned_cv),
                    "params": tuned_result['params'],
                    "metric": metric_name,
                    "model_obj": tuned_result['model']
                }

                if tuned_cv > best_cv_score:
                    log(f"[MODEL] ✓ Tuning improved CV: {best_cv_score:.4f} → {tuned_cv:.4f}")
                    best_cv_score = tuned_cv
                    best_model_name = tuned_result['name']
                    best_model_obj = tuned_result['model']

        # --- Create ensemble from TRAINED model objects ---
        ensemble_members = []
        ensemble, ensemble_members = create_ensemble(results, X_train_use, y_train_use, problem_type, log_callback)

        if ensemble:
            # Keep selection metric consistent with base models (CV-based).
            try:
                scoring = 'accuracy' if is_classification else 'r2'
                ensemble_cv_scores = cross_val_score(ensemble, X_train_use, y_train_use, cv=cv_strategy, scoring=scoring)
                ensemble_cv = float(ensemble_cv_scores.mean())
            except Exception:
                ensemble_cv = -np.inf

            y_pred_ensemble = ensemble.predict(X_test_use)

            # Inverse transform if target was transformed
            if target_transformer is not None and not is_classification:
                y_pred_ensemble = target_transformer.inverse_transform(y_pred_ensemble.reshape(-1, 1)).ravel()

            if is_classification:
                ensemble_score = accuracy_score(y_test, y_pred_ensemble)
                avg = 'binary' if len(y.unique()) == 2 else 'weighted'
                e_prec = precision_score(y_test, y_pred_ensemble, average=avg, zero_division=0)
                e_rec = recall_score(y_test, y_pred_ensemble, average=avg, zero_division=0)
                e_f1 = f1_score(y_test, y_pred_ensemble, average=avg, zero_division=0)
                e_auc = None
                try:
                    if hasattr(ensemble, 'predict_proba'):
                        y_proba_e = ensemble.predict_proba(X_test_use)
                        if len(y.unique()) == 2:
                            e_auc = roc_auc_score(y_test, y_proba_e[:, 1])
                        else:
                            e_auc = roc_auc_score(y_test, y_proba_e, multi_class='ovr', average='weighted')
                except Exception:
                    pass
                auc_str = f", AUC={e_auc:.4f}" if e_auc is not None else ""
                log(f"[MODEL] Ensemble: Acc={ensemble_score:.4f}, F1={e_f1:.4f}{auc_str}")
                results['ensemble'] = {
                    "score": float(ensemble_score),
                    "accuracy": float(ensemble_score),
                    "precision": float(e_prec),
                    "recall": float(e_rec),
                    "f1_score": float(e_f1),
                    "auc_roc": float(e_auc) if e_auc is not None else None,
                    "cv_score": None if ensemble_cv == -np.inf else float(ensemble_cv),
                    "metric": metric_name
                }
            else:
                ensemble_score = r2_score(y_test, y_pred_ensemble)
                n = len(y_test)
                p = X_test_use.shape[1]
                adj_r2 = 1 - (1 - ensemble_score) * (n - 1) / (n - p - 1) if n > p + 1 else ensemble_score
                mse = mean_squared_error(y_test, y_pred_ensemble)
                rmse = mse ** 0.5
                mae = mean_absolute_error(y_test, y_pred_ensemble)
                log(f"[MODEL] Ensemble: R²={ensemble_score:.4f}, RMSE={rmse:.4f}")
                results['ensemble'] = {
                    "score": float(ensemble_score),
                    "r2": float(ensemble_score),
                    "adjusted_r2": float(adj_r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "cv_score": None if ensemble_cv == -np.inf else float(ensemble_cv),
                    "metric": metric_name
                }

            # Use the same selection criterion as base models: CV score.
            if ensemble_cv > best_cv_score:
                log(f"[MODEL] ✓ Ensemble beats best! Using ensemble.")
                best_model_name = 'ensemble'
                best_cv_score = ensemble_cv

        # Get the final best test score for reporting
        best_score = results.get(best_model_name, {}).get("score", 0)
        log(f"[MODEL] 🏆 Final: {best_model_name} (test={best_score:.4f})")

        # --- SHAP Feature Importance (using SCALED test data to match training) ---
        feature_importance = None
        if SHAP_AVAILABLE:
            shap_model = best_model_obj
            if best_model_name == 'ensemble':
                # SHAP needs a tree-based model — find the best one by CV score
                tree_model_names = {'random_forest', 'xgboost', 'gradient_boosting', 'lightgbm',
                                    'decision_tree', 'random_forest_tuned', 'xgboost_tuned',
                                    'gradient_boosting_tuned'}
                shap_model = None
                for name, info in sorted(results.items(), key=lambda x: x[1].get("cv_score", 0), reverse=True):
                    if name in tree_model_names and info.get("model_obj"):
                        shap_model = info["model_obj"]
                        log(f"[MODEL] Using {name} for SHAP analysis (tree-based)")
                        break
                if shap_model is None:
                    log(f"[MODEL] No tree-based model found for SHAP, skipping.")

            if shap_model:
                feature_importance = analyze_feature_importance(
                    shap_model,
                    X_test_scaled,  # FIXED: use scaled data (matches training)
                    X_train.columns.tolist(),
                    output_folder=output_folder,
                    log_callback=log_callback
                )

        log(f"[MODEL] ✓ Training complete!")

        # Get params for the best model
        final_best_params = results.get(best_model_name, {}).get("params", {})
        if best_model_name == 'ensemble':
            final_best_params = {"ensemble_voting": "soft" if is_classification else "average"}

        # Clean model_obj references from results before returning (not serializable)
        clean_results = {}
        for name, info in results.items():
            clean_results[name] = {k: v for k, v in info.items() if k != "model_obj"}

        return A2AResponse(
            task_id=task.task_id,
            sender="model-agent",
            status="COMPLETED",
            output={
                "models": clean_results,
                "best_model": best_model_name,
                "best_score": float(best_score),
                "best_params": final_best_params,
                "ensemble_members": ensemble_members if best_model_name == 'ensemble' else [],
                "problem_type": problem_type,
                "metric": metric_name,
                "used_tuning": tuned_result is not None,
                "used_balancing": actually_balanced,  # FIXED: reflects actual usage
                "used_scaling": True,
                "target_transform": not is_classification and target_transformer is not None,
                "feature_importance": feature_importance[:10] if feature_importance else None
            }
        )
    except Exception as e:
        log(f"[MODEL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise



