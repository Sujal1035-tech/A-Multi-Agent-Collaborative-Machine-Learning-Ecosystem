import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Suppress warnings
warnings.filterwarnings('ignore')
# --- Configuration ---
DATA_FILE = 'data.csv'
TARGET_COLUMN = 'Region'
CATEGORICAL_COLS = ['Classes'] 
def setup_directories():
    """Creates the necessary output directories (stats, plots, reports) at the root level."""
    print("Setting up required directories (stats, plots, reports)...")
    try:
        os.makedirs('stats', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        print("Directories created successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not create directories: {e}")
        sys.exit(1)
def load_and_preprocess_data(file_path):
    """Loads data, performs cleaning, and applies LabelEncoding to categorical columns."""
    print(f"\nLoading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file {file_path} not found.")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    print(f"Initial data shape: {df.shape}")
    print("\nData Head:")
    print(df.head())
    # --- Preprocessing ---
    print("\nStarting preprocessing steps...")
    # Handle Categorical Columns (Critical Requirement: Use LabelEncoder)
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].dtype == 'object':
            # Step 1: Clean data (strip whitespace, common issue in object columns)
            df[col] = df[col].astype(str).str.strip()
            # Step 2: Apply Label Encoding
            df[col] = le.fit_transform(df[col])
            print(f"Column '{col}' cleaned and LabelEncoded successfully.")
        elif col in df.columns:
            # If it's supposed to be categorical but isn't object type (e.g., already encoded)
            print(f"Column '{col}' is already numeric ({df[col].dtype}), skipping LabelEncoder.")
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
        return None, None, None
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y, df
def perform_eda(df):
    """Generates basic visualizations and saves them to the plots/ directory."""
    print("\nPerforming Exploratory Data Analysis (EDA)...")
    try:
        # Plot 1: Target Variable Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x=TARGET_COLUMN, data=df)
        plt.title(f'Distribution of {TARGET_COLUMN}')
        plt.savefig('plots/target_distribution.png')
        plt.close()
        print("- Saved plots/target_distribution.png")
        # Plot 2: Correlation Heatmap
        # Ensure only numeric columns are included for correlation
        numeric_df = df.select_dtypes(include=np.number)
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap')
        plt.savefig('plots/correlation_heatmap.png')
        plt.close()
        print("- Saved plots/correlation_heatmap.png")
    except Exception as e:
        print(f"Error during EDA: {e}")
def train_and_evaluate_models(X, y):
    """Trains multiple classifiers and saves reports and performance stats."""
    print("\nStarting Model Training and Evaluation...")
    # Ensure consistent splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    models = {
        # CRITICAL: max_iter=1000 required for Logistic Regression
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        # Best model identified
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) 
    }
    performance_data = {}
    for name, model in models.items():
        try:
            print(f"--- Training {name} ---")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # CRITICAL: zero_division=0 required
            report = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            performance_data[name] = accuracy
            # Save Classification Report
            report_filename = f'reports/{name}_classification_report.txt'
            with open(report_filename, 'w') as f:
                f.write(f"Classification Report for {name}:\n\n")
                f.write(report)
            print(f"-> Accuracy: {accuracy:.4f}. Report saved to {report_filename}")
            # Save Confusion Matrix Plot
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(f'plots/{name}_confusion_matrix.png')
            plt.close()
        except Exception as e:
            print(f"Error during training or evaluation of {name}: {e}")
    # Save Model Performance Summary (CRITICAL REQUIREMENT: stats/model_performance.txt)
    save_performance_summary(performance_data)
def save_performance_summary(performance_data):
    """Saves the accuracy scores to stats/model_performance.txt."""
    performance_path = 'stats/model_performance.txt'
    print(f"\nSaving model performance summary to {performance_path}...")
    try:
        # Sort results by accuracy score (descending)
        sorted_data = sorted(performance_data.items(), key=lambda item: item[1], reverse=True)
        with open(performance_path, 'w') as f:
            f.write("--- Model Performance Summary (Accuracy Scores) ---\n")
            for name, score in sorted_data:
                f.write(f"{name}: {score:.4f}\n")
        print("Model performance summary saved successfully.")
    except Exception as e:
        print(f"Error saving performance data: {e}")
def main():
    """Main execution function."""
    setup_directories()
    X, y, df = load_and_preprocess_data(DATA_FILE)
    if X is None or y is None:
        print("Analysis terminated due to data loading/preprocessing failure.")
        return
    perform_eda(df)
    train_and_evaluate_models(X, y)
    print("\n--- Analysis Complete ---")
if __name__ == '__main__':
    main()