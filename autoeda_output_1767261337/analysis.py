import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# CRITICAL REQUIREMENT: Suppress warnings for cleaner output and handling specific model params
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) # For XGBoost warnings
DATA_FILE = 'data.csv'
def setup_directories():
    """Creates required output directories directly at the root level."""
    print("--- Setting up directories ---")
    try:
        # Create directories required by the task
        os.makedirs('stats', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        print("Directories created successfully: stats/, plots/, reports/")
    except Exception as e:
        print(f"Error creating directories: {e}")
        # Raising the error to halt execution if setup fails
        raise
def load_data(file_path):
    """Loads data and prints a summary."""
    try:
        df = pd.read_csv(file_path)
        print(f"\n--- Data Loading ---")
        print(f"Data loaded successfully. Shape: {df.shape}")
        print("Data head:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Critical Error: Input file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error during data loading: {e}")
        return None
def preprocess_data(df):
    """Applies Label Encoding to the target column 'species' and prepares X and y."""
    if 'species' not in df.columns:
        raise ValueError("Target column 'species' not found in data.")
    # CRITICAL REQUIREMENT: Use LabelEncoder for categorical columns
    le = LabelEncoder()
    df['species_encoded'] = le.fit_transform(df['species'])
    class_names = le.classes_
    print(f"\n--- Preprocessing ---")
    print(f"Target variable 'species' encoded. Class labels: {class_names}")
    # Features (X) exclude original target and encoded target
    X = df.drop(['species', 'species_encoded'], axis=1)
    # Target (y) is the encoded column
    y = df['species_encoded']
    return X, y, class_names
def perform_eda(df_raw):
    """Generates and saves basic EDA visualizations."""
    print("\n--- Starting EDA ---")
    if 'species' not in df_raw.columns:
        print("Skipping EDA: Target column 'species' is missing.")
        return
    try:
        # 1. Distribution of numerical features (Histograms)
        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(num_cols):
            plt.subplot(2, 2, i + 1)
            sns.histplot(df_raw[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig('plots/numerical_feature_distributions.png')
        plt.close()
        # 2. Target distribution (Bar plot)
        plt.figure(figsize=(7, 5))
        sns.countplot(y=df_raw['species'])
        plt.title('Target Class Distribution (Species)')
        plt.savefig('plots/target_class_distribution.png')
        plt.close()
        print("EDA visualizations saved to plots/ directory.")
    except Exception as e:
        print(f"Error during EDA visualization: {e}")
def train_and_evaluate_models(X, y, class_names):
    """Trains various classification models and saves performance reports."""
    print("\n--- Model Training and Evaluation ---")
    # Stratified split for balanced test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    models = {
        # CRITICAL REQUIREMENT: max_iter=1000 for LogisticRegression
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42, 
            n_estimators=100
        )
    }
    performance_metrics = []
    for name, model in models.items():
        try:
            print(f"Training and evaluating: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            performance_metrics.append({
                'Model': name,
                'Accuracy': accuracy
            })
            # --- Save Classification Report ---
            # CRITICAL REQUIREMENT: zero_division=0
            report = classification_report(
                y_test, y_pred, target_names=class_names, zero_division=0
            )
            report_path = f'reports/{name}_classification_report.txt'
            with open(report_path, 'w') as f:
                f.write(f"Classification Report for {name} (Target: {class_names}):\n\n")
                f.write(report)
            # --- Save Confusion Matrix Plot ---
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names
            )
            plt.title(f'Confusion Matrix for {name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(f'plots/{name}_confusion_matrix.png')
            plt.close()
        except Exception as e:
            print(f"Error encountered during training or evaluation of {name}: {e}")
            performance_metrics.append({'Model': name, 'Accuracy': 'ERROR'})
    # --- Save Model Performance Summary (CRITICAL REQUIREMENT) ---
    performance_df = pd.DataFrame(performance_metrics)
    performance_summary_path = 'stats/model_performance.txt'
    try:
        with open(performance_summary_path, 'w') as f:
            f.write("Model Performance Summary\n")
            f.write("=" * 30 + "\n")
            f.write(performance_df.to_string(index=False, float_format="%.4f"))
            f.write("\n\nNote: All models trained using Label Encoding on the target variable 'species'.\n")
        print(f"\nModel performance summary saved to {performance_summary_path}")
    except Exception as e:
        print(f"Error saving model performance summary: {e}")
def main():
    try:
        setup_directories()
        # 1. Load Data
        df_raw = load_data(DATA_FILE)
        if df_raw is None:
            return
        # 2. EDA (using original dataframe)
        perform_eda(df_raw.copy())
        # 3. Preprocessing (Encoding and Splitting)
        X, y, class_names = preprocess_data(df_raw)
        # 4. Modeling
        train_and_evaluate_models(X, y, class_names)
        print("\n--- Analysis complete. Outputs saved to stats/, plots/, and reports/ directories. ---")
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] An unhandled error occurred during the overall analysis pipeline: {e}")
if __name__ == '__main__':
    main()