from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from a2a.schemas import A2ATask, A2AResponse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Smart ML imports
try:
    import optuna
    from imblearn.over_sampling import SMOTE
    import shap
    SMART_ML_AVAILABLE = True
except ImportError:
    SMART_ML_AVAILABLE = False
    print("[MODEL] Warning: optuna, imbalanced-learn, or shap not installed. Smart features disabled.")


app = FastAPI()

def preprocess_data(df, prep_strategy, feat_strategy, target_col):
    """Apply preprocessing and feature engineering"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])
    
    return X, y

def handle_class_imbalance(X, y):
    """Smart imbalance handling with SMOTE"""
    if not SMART_ML_AVAILABLE:
        return X, y
    
    # Check imbalance ratio
    value_counts = y.value_counts()
    if len(value_counts) < 2:
        return X, y
    
    ratio = value_counts.min() / value_counts.max()
    
    if ratio < 0.5:  # Imbalanced
        print(f"[MODEL] ⚠️  Class imbalance detected (ratio: {ratio:.2f})")
        print(f"[MODEL] 🔧 Applying SMOTE to balance classes...")
        
        try:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"[MODEL] ✅ Balanced: {len(X)} → {len(X_balanced)} samples")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"[MODEL] ⚠️  SMOTE failed: {e}. Continuing without balancing.")
            return X, y
    
    return X, y

def hyperparameter_tuning(X_train, y_train, X_test, y_test, problem_type):
    """Optuna-based hyperparameter optimization"""
    if not SMART_ML_AVAILABLE:
        return None
    
    print(f"[MODEL] 🔍 Starting hyperparameter tuning with Optuna...")
    
    def objective(trial):
        # Try different model types
        model_type = trial.suggest_categorical('model', ['random_forest', 'xgboost'])
        
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            
            if problem_type == 'classification':
                model = RandomForestClassifier(**params, random_state=42)
            else:
                model = RandomForestRegressor(**params, random_state=42)
        
        else:  # xgboost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            
            if problem_type == 'classification':
                model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
            else:
                model = XGBRegressor(**params, random_state=42)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if problem_type == 'classification':
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)
        
        return score
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    
    print(f"[MODEL] ✅ Best score with tuning: {study.best_value:.4f}")
    print(f"[MODEL] 📊 Best params: {study.best_params}")
    
    # Train best model
    best_params = study.best_params
    model_type = best_params.pop('model')
    
    if model_type == 'random_forest':
        if problem_type == 'classification':
            best_model = RandomForestClassifier(**best_params, random_state=42)
        else:
            best_model = RandomForestRegressor(**best_params, random_state=42)
    else:
        if problem_type == 'classification':
            best_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            best_model = XGBRegressor(**best_params, random_state=42)
    
    best_model.fit(X_train, y_train)
    
    return {
        'model': best_model,
        'score': study.best_value,
        'params': study.best_params,
        'name': f'{model_type}_tuned'
    }

def create_ensemble(models_dict, X_train, y_train, problem_type):
    """Create voting ensemble from top models"""
    print(f"[MODEL] 🎯 Creating ensemble from top models...")
    
    # Get top 3 models
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    
    estimators = []
    for name, score in sorted_models:
        if name == 'logistic_regression' or name == 'linear_regression':
            continue  # Skip simple models for ensemble
        
        if problem_type == 'classification':
            if 'random_forest' in name:
                estimators.append((name, RandomForestClassifier(n_estimators=100, random_state=42)))
            elif 'xgboost' in name:
                estimators.append((name, XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')))
            elif 'decision_tree' in name:
                estimators.append((name, DecisionTreeClassifier(random_state=42)))
        else:
            if 'random_forest' in name:
                estimators.append((name, RandomForestRegressor(n_estimators=100, random_state=42)))
            elif 'xgboost' in name:
                estimators.append((name, XGBRegressor(n_estimators=100, random_state=42)))
            elif 'decision_tree' in name:
                estimators.append((name, DecisionTreeRegressor(random_state=42)))
    
    if len(estimators) < 2:
        return None
    
    if problem_type == 'classification':
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
    else:
        from sklearn.ensemble import VotingRegressor
        ensemble = VotingRegressor(estimators=estimators)
    
    ensemble.fit(X_train, y_train)
    print(f"[MODEL] ✅ Ensemble created from {len(estimators)} models")
    
    return ensemble

def analyze_feature_importance(best_model, X_test, feature_names, output_folder='plots'):
    """Analyze and visualize feature importance using SHAP"""
    if not SMART_ML_AVAILABLE or not hasattr(shap, 'TreeExplainer'):
        print("[MODEL] ⚠️  SHAP not available, skipping feature importance analysis")
        return None
    
    try:
        print(f"[MODEL] 🔍 Analyzing feature importance with SHAP...")
        
        # Create output folder if needed
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        # Check if model is tree-based
        tree_models = (RandomForestClassifier, RandomForestRegressor, 
                      XGBClassifier, XGBRegressor,
                      DecisionTreeClassifier, DecisionTreeRegressor)
        
        if not isinstance(best_model, tree_models):
            print("[MODEL] ⚠️  Model is not tree-based, skipping SHAP analysis")
            return None
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle multi-class output (take first class for simplicity)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Calculate feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        print(f"[MODEL] 📊 Top 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"[MODEL]   {row['feature']}: {row['importance']:.4f}")
        
        # Save summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=15)
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
        
        print(f"[MODEL] ✅ SHAP plots saved to {output_folder}/")
        
        return importance_df.to_dict('records')
        
    except Exception as e:
        print(f"[MODEL] ⚠️  SHAP analysis failed: {e}")
        return None

@app.post("/a2a")
def handle(task: A2ATask):
    try:
        print(f"[MODEL] Starting SMART model training...")
        csv_path = task.input["csv_path"]
        prep_strategy = task.input.get("prep_strategy", {})
        feat_strategy = task.input.get("feat_strategy", {})
        
        print(f"[MODEL] Loading dataset...")
        df = pd.read_csv(csv_path)
        target_col = df.columns[-1]
        
        print(f"[MODEL] Preprocessing data...")
        X, y = preprocess_data(df, prep_strategy, feat_strategy, target_col)
        print(f"[MODEL] Data shape: X={X.shape}, y={y.shape}")
        
        # Detect problem type
        is_classification = len(y.unique()) < 20 and y.dtype in ['int64', 'object']
        problem_type = "classification" if is_classification else "regression"
        print(f"[MODEL] Problem type: {problem_type}")
        
        # Handle class imbalance (Step 1: Quick Win)
        if is_classification and SMART_ML_AVAILABLE:
            X, y = handle_class_imbalance(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"[MODEL] Train/test split: {len(X_train)}/{len(X_test)}")
        
        # Define base models
        if is_classification:
            models = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "decision_tree": DecisionTreeClassifier(random_state=42),
                "xgboost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
            }
            metric_name = "accuracy"
        else:
            models = {
                "linear_regression": LinearRegression(),
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "decision_tree": DecisionTreeRegressor(random_state=42),
                "xgboost": XGBRegressor(n_estimators=100, random_state=42)
            }
            metric_name = "r2_score"
        
        print(f"[MODEL] Training {len(models)} base models with cross-validation...")
        
        # Train base models with cross-validation (Step 2: Cross-validation)
        results = {}
        best_score = -np.inf
        best_model_name = None
        
        for i, (name, model) in enumerate(models.items(), 1):
            print(f"[MODEL] [{i}/{len(models)}] Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy' if is_classification else 'r2')
            cv_mean = cv_scores.mean()
            
            # Train on full training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classification:
                test_score = accuracy_score(y_test, y_pred)
            else:
                test_score = r2_score(y_test, y_pred)
            
            print(f"[MODEL] [{i}/{len(models)}] {name}: test={test_score:.4f}, cv_mean={cv_mean:.4f}")
            
            results[name] = {
                "score": float(test_score), 
                "cv_score": float(cv_mean),
                "metric": metric_name
            }
            
            if test_score > best_score:
                best_score = test_score
                best_model_name = name
        
        print(f"[MODEL] Best base model: {best_model_name} ({best_score:.4f})")
        
        # Conditional Hyperparameter Tuning (if accuracy < 80%)
        tuned_result = None
        if best_score < 0.80 and SMART_ML_AVAILABLE:
            print(f"[MODEL] ⚠️  Accuracy below 80% ({best_score:.4f})")
            print(f"[MODEL] 🚀 Triggering hyperparameter tuning...")
            tuned_result = hyperparameter_tuning(X_train, y_train, X_test, y_test, problem_type)
            
            if tuned_result and tuned_result['score'] > best_score:
                print(f"[MODEL] ✅ Tuning improved score: {best_score:.4f} → {tuned_result['score']:.4f}")
                best_score = tuned_result['score']
                best_model_name = tuned_result['name']
                results[tuned_result['name']] = {
                    "score": float(tuned_result['score']),
                    "params": tuned_result['params'],
                    "metric": metric_name
                }
        
        # Create ensemble (Step 3: Ensemble)
        ensemble = create_ensemble(
            {k: v['score'] for k, v in results.items()},
            X_train, y_train,
            problem_type
        )
        
        if ensemble:
            y_pred_ensemble = ensemble.predict(X_test)
            if is_classification:
                ensemble_score = accuracy_score(y_test, y_pred_ensemble)
            else:
                ensemble_score = r2_score(y_test, y_pred_ensemble)
            
            print(f"[MODEL] 🎯 Ensemble score: {ensemble_score:.4f}")
            
            results['ensemble'] = {
                "score": float(ensemble_score),
                "metric": metric_name
            }
            
            if ensemble_score > best_score:
                print(f"[MODEL] ✅ Ensemble is best! {best_score:.4f} → {ensemble_score:.4f}")
                best_score = ensemble_score
                best_model_name = 'ensemble'
        
        print(f"[MODEL] 🏆 Final best model: {best_model_name} ({best_score:.4f})")
        
        # Feature Importance Analysis with SHAP
        feature_importance = None
        if best_model_name in models and SMART_ML_AVAILABLE:
            best_model_obj = models.get(best_model_name) or \
                            (tuned_result['model'] if tuned_result and best_model_name == tuned_result['name'] else None) or \
                            (ensemble if best_model_name == 'ensemble' else None)
            
            if best_model_obj:
                feature_importance = analyze_feature_importance(
                    best_model_obj,
                    X_test,
                    X.columns.tolist(),
                    output_folder='plots'
                )
        
        print(f"[MODEL] Training complete!")
        
        return A2AResponse(
            task_id=task.task_id,
            sender="model-agent",
            status="COMPLETED",
            output={
                "models": results,
                "best_model": best_model_name,
                "best_score": float(best_score),
                "problem_type": problem_type,
                "metric": metric_name,
                "used_tuning": tuned_result is not None,
                "used_balancing": SMART_ML_AVAILABLE,
                "feature_importance": feature_importance[:10] if feature_importance else None  # Top 10
            }
        )
    except Exception as e:
        print(f"[MODEL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
