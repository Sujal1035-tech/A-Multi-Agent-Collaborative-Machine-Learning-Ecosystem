import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import xgboost as xgb
warnings.filterwarnings('ignore')
# Create directories
os.makedirs('stats', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)
try:
    # Load data
    data = pd.read_csv('data.csv')
    print("Data loaded successfully.")
    print(data.head())
    # --- Preprocessing ---
    # Handle missing values in 'horsepower' using mean imputation
    data['horsepower'] = data['horsepower'].fillna(data['horsepower'].mean())
    # Encode categorical features
    categorical_cols = ['origin', 'name']
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    print("\nCategorical features encoded.")
    # Split data
    X = data.drop('mpg', axis=1)
    y = data['mpg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nData split into training and testing sets.")
    # --- EDA and Visualization ---
    # Histograms for numerical features
    numerical_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'plots/{col}_histogram.png')
        plt.close()
    print("\nHistograms generated and saved.")
    # Bar charts for categorical features
    for col in ['origin']:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=data[col])
        plt.title(f'Distribution of {col}')
        plt.savefig(f'plots/{col}_barplot.png')
        plt.close()
    print("\nBar plots generated and saved.")
    # --- Model Training and Evaluation ---
    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_predictions)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
    # Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    # Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_r2 = r2_score(y_test, dt_predictions)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
    # XGBoost Regressor
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_predictions)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    # Save model performance
    with open('stats/model_performance.txt', 'w') as f:
        f.write("Gradient Boosting Regressor:\n")
        f.write(f"  R2 Score: {gb_r2:.4f}\n")
        f.write(f"  RMSE: {gb_rmse:.4f}\n\n")
        f.write("Random Forest Regressor:\n")
        f.write(f"  R2 Score: {rf_r2:.4f}\n")
        f.write(f"  RMSE: {rf_rmse:.4f}\n\n")
        f.write("Decision Tree Regressor:\n")
        f.write(f"  R2 Score: {dt_r2:.4f}\n")
        f.write(f"  RMSE: {dt_rmse:.4f}\n\n")
        f.write("XGBoost Regressor:\n")
        f.write(f"  R2 Score: {xgb_r2:.4f}\n")
        f.write(f"  RMSE: {xgb_rmse:.4f}\n")
    print("\nModel performance saved to stats/model_performance.txt")
    print("\nAll tasks completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")