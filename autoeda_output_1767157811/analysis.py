import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# XGBoost is crucial for comprehensive analysis
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Skipping XGBoost model training.")
    XGB_AVAILABLE = False
# --- Configuration ---
DATA_FILE = 'data.csv'
# Suppress specific warnings for cleaner output (e.g., convergence warnings)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
def setup_environment():
    """Creates necessary output directories directly at the root level."""
    print("--- 1. Setting up Environment ---")
    try:
        # CRITICAL REQUIREMENT: Create directories directly at root level
        os.makedirs('stats', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        print("Directories created successfully: stats/, plots/, reports/")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not create necessary directories: {e}")
        exit(1)
def load_data(file_path):
    """Loads data and performs initial preprocessing (Label Encoding for target)."""
    print(f"\n--- 2. Loading and Preprocessing Data ({file_path}) ---")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print("Data head:")
        print(df.head())
        # CRITICAL REQUIREMENT: Preprocess categorical columns using LabelEncoder
        le = LabelEncoder()
        # Identify categorical/object columns (assuming 'species' is the main one)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if not categorical_cols:
            print("No object type categorical columns found for encoding.")
        for col in categorical_cols:
            if col in df.columns:
                print(f"Encoding categorical column: '{col}'")
                # Fit and transform the column
                df[col] = le.fit_transform(df[col])
        return df, le
    except FileNotFoundError:
        print(f"\nCRITICAL ERROR: Data file '{file_path}' not found.")
        print("Please ensure 'data.csv' is present in the execution directory.")
        return None, None
    except Exception as e:
        print(f"Error during data loading or initial preprocessing: {e}")
        return None, None
def run_eda(df):
    """Generates and saves exploratory data analysis plots."""
    print("\n--- 3. Running Exploratory Data Analysis (EDA) ---")
    try:
        # 1. Histograms of numerical features
        numerical_cols = [col for col in df.columns if col != 'species']
        df[numerical_cols].hist(figsize=(10, 8))
        plt.suptitle("Feature Distributions (Histograms)", y=1.02)
        plt.tight_layout()
        plt.savefig('plots/feature_histograms.png')
        plt.close()
        print("Saved: plots/feature_histograms.png")
        # 2. Correlation Heatmap
        plt.figure(figsize=(8, 7))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap')
        plt.savefig('plots/correlation_heatmap.png')
        plt.close()
        print("Saved: plots/correlation_heatmap.png")
    except Exception as e:
        print(f"Error during EDA visualization: {e}")
def train_and_evaluate_models(df, label_encoder):
    """Trains classification models, generates metrics, and saves outputs."""
    print("\n--- 4. Model Training and Evaluation ---")
    try:
        X = df.drop('species', axis=1)
        y = df['species']
        # Stratified split to ensure class balance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        models = {
            # CRITICAL: Use max_iter=1000 for Logistic Regression
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42), 
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
        }
        if XGB_AVAILABLE:
            # XGBoost requires target labels starting from 0 (which LabelEncoder provides)
            models['XGBoost'] = XGBClassifier(
                use_label_encoder=False, 
                eval_metric='mlogloss', 
                random_state=42,
                n_estimators=100
            )
        performance_results = {}
        target_names = label_encoder.classes_
        for name, model in models.items():
            print(f"\n-> Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # --- Evaluation ---
            accuracy = accuracy_score(y_test, y_pred)
            performance_results[name] = accuracy
            # 1. Classification Report (Saved to reports/)
            # CRITICAL: Use zero_division=0
            report = classification_report(y_test, y_pred, zero_division=0, 
                                           target_names=target_names)
            report_path = f'reports/{name}_classification_report.txt'
            with open(report_path, 'w') as f:
                f.write(f"Classification Report for {name}\n")
                f.write("=" * 40 + "\n")
                f.write(report)
                f.write(f"\nAccuracy: {accuracy:.4f}")
            print(f"Accuracy: {accuracy:.4f} | Saved report to {report_path}")
            # 2. Confusion Matrix (Saved to plots/)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            cm_path = f'plots/{name}_confusion_matrix.png'
            plt.savefig(cm_path)
            plt.close()
            print(f"Saved confusion matrix to {cm_path}")
        # 3. Aggregate Performance Results (Saved to stats/model_performance.txt)
        stats_path = 'stats/model_performance.txt'
        with open(stats_path, 'w') as f:
            f.write("--- Model Performance Summary ---\n")
            f.write("Note: Based on 30% Test Split\n\n")
            # Sort by accuracy descending
            sorted_results = sorted(performance_results.items(), key=lambda item: item[1], reverse=True)
            for name, score in sorted_results:
                f.write(f"{name:<20}: Accuracy = {score:.4f}\n")
        print(f"\nSuccessfully saved model performance summary to {stats_path}")
    except Exception as e:
        print(f"\nCRITICAL ERROR during model training/evaluation: {e}")
def main():
    """Main pipeline execution."""
    # 1. Setup Environment
    setup_environment()
    # 2. Load Data and Preprocess
    df, le = load_data(DATA_FILE)
    if df is None:
        print("\nAnalysis pipeline halted.")
        return
    # 3. Run EDA
    run_eda(df)
    # 4. Train and Evaluate Models
    train_and_evaluate_models(df, le)
    print("\nAnalysis complete.")
if __name__ == "__main__":
    main()