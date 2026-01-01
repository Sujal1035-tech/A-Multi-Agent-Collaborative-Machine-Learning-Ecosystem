import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression 
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
np.random.seed(42)
def create_directories():
    """Creates required output directories: stats, plots, reports."""
    try:
        print("Creating required directories: stats/, plots/, reports/...")
        # Create directories DIRECTLY at root level
        os.makedirs('stats', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
def load_data(file_path='data.csv'):
    """Loads the dataset."""
    try:
        # CRITICAL REQUIREMENT 1: Use 'data.csv'
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print("\nInitial Data Head:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please ensure it is in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def preprocess_data(df):
    """
    Handles missing values (imputation), encodes categorical variables using LabelEncoder,
    and prepares data for modeling.
    """
    print("\nStarting Preprocessing...")
    # 1. Handle Type Conversion and Missing Values
    try:
        # Convert 'horsepower' to numeric, handling potential non-numeric entries ('?')
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        # Impute missing 'horsepower' (1.51% missing) with the median
        imputer = SimpleImputer(strategy='median')
        df['horsepower'] = imputer.fit_transform(df[['horsepower']])
        print("- Missing values in 'horsepower' imputed with median.")
    except Exception as e:
        print(f"Warning: Could not impute horsepower: {e}")
    # 2. Identify Categorical Columns
    categorical_cols = ['origin', 'name']
    # 3. Encode Categorical Columns using LabelEncoder (CRITICAL REQUIREMENT 2)
    le = LabelEncoder()
    for col in categorical_cols:
        try:
            df[col] = df[col].astype(str) 
            df[col] = le.fit_transform(df[col])
            print(f"- Column '{col}' encoded using LabelEncoder.")
        except Exception as e:
            print(f"Error encoding column {col}: {e}")
    # 4. Feature and Target Definition
    target_column = 'mpg'
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]
    return X, y
def perform_eda(df):
    """Generates and saves basic EDA plots to the 'plots/' directory."""
    print("\nPerforming EDA and saving plots to 'plots/'...")
    try:
        # 1. Target Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df['mpg'], kde=True)
        plt.title('Distribution of MPG (Target)')
        plt.savefig('plots/01_mpg_distribution.png')
        plt.close()
        # 2. Correlation Heatmap
        numerical_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap')
        plt.savefig('plots/02_correlation_heatmap.png')
        plt.close()
        print("- Saved 2 EDA plots to 'plots/'.")
    except Exception as e:
        print(f"Error during EDA: {e}")
def save_regression_report(model_name, y_test, y_pred, metrics):
    """Saves a detailed regression report to the 'reports/' directory."""
    try:
        report_path = f"reports/{model_name.lower().replace(' ', '_')}_regression_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"----- Detailed Regression Report: {model_name} -----\n")
            f.write(f"R2 Score: {metrics['R2 Score']:.4f}\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}\n")
            f.write(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}\n")
            residuals = y_test - y_pred
            f.write("\n----- Residuals Summary -----\n")
            f.write(f"Mean Residual: {residuals.mean():.4f}\n")
            f.write(f"Std Dev Residual: {residuals.std():.4f}\n")
        print(f"- Saved detailed report for {model_name} to 'reports/'.")
    except Exception as e:
        print(f"Error saving regression report for {model_name}: {e}")
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains regression models and evaluates them."""
    # Using regression models as target is 'mpg' (continuous)
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGB Regressor': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    results = {}
    print("\nStarting Model Training and Evaluation...")
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            metrics = {
                'R2 Score': r2,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            }
            results[name] = metrics
            print(f"  {name} R2 Score: {r2:.4f}")
            # Save detailed report for the highest-performing model type (Random Forest)
            if name == 'Random Forest Regressor':
                 save_regression_report(name, y_test, y_pred, metrics)
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {'R2 Score': -999, 'MAE': -999, 'MSE': -999, 'RMSE': -999}
    return results
def save_model_performance(results):
    """Saves the summary of model performance to stats/model_performance.txt (CRITICAL REQUIREMENT 7)."""
    output_path = 'stats/model_performance.txt'
    print(f"\nSaving model performance to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            f.write("--- Model Performance Summary (Regression) ---\n\n")
            # Sort models by R2 score (descending)
            sorted_results = sorted(results.items(), key=lambda item: item[1]['R2 Score'], reverse=True)
            for name, metrics in sorted_results:
                f.write(f"Model: {name}\n")
                f.write(f"  R2 Score: {metrics['R2 Score']:.4f}\n")
                f.write(f"  MAE: {metrics['MAE']:.4f}\n")
                f.write(f"  MSE: {metrics['MSE']:.4f}\n")
                f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
                f.write("-" * 30 + "\n")
        print("Model performance successfully saved to stats/model_performance.txt.")
    except Exception as e:
        print(f"Error saving model performance statistics: {e}")
# --- Main Execution ---
def main():
    try:
        # 1. Setup Directories
        create_directories()
        # 2. Load Data
        df = load_data('data.csv')
        if df is None:
            return
        # 3. EDA 
        perform_eda(df.copy())
        # 4. Preprocessing & Splitting
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"\nTraining set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
        # 5. Model Training and Evaluation
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        # 6. Reporting
        save_model_performance(results)
        print("\nProduction analysis complete.")
    except Exception as e:
        # Catch any high-level errors
        print(f"\nFATAL ERROR during analysis execution: {e}")
if __name__ == '__main__':
    main()