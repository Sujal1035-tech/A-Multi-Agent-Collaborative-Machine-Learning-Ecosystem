"""
AutoEDA Generated Analysis
This code reproduces the exact pipeline that the agents performed.
Best Model: ensemble
Problem Type: regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Model Import
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ============================================
# 1. SETUP
# ============================================
os.makedirs('stats', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ============================================
# 2. LOAD DATA
# ============================================
try:
    data = pd.read_csv('data.csv')
    print(f'Data loaded. Shape: {data.shape}')
except Exception as e:
    print(f'Error loading data: {e}')
    exit(1)

# ============================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================
TARGET_COL = 'price'
X = data.drop(TARGET_COL, axis=1)
y = data[TARGET_COL]

# Encode target if categorical
target_le = None
if y.dtype == 'object' or str(y.dtype) == 'category':
    target_le = LabelEncoder()
    y = pd.Series(target_le.fit_transform(y), index=y.index)
    print(f'Encoded target: {dict(zip(target_le.classes_, range(len(target_le.classes_))))}')

# ============================================
# 4. TRAIN/TEST SPLIT (BEFORE preprocessing to prevent data leakage)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# ============================================
# 5. PREPROCESSING (fit on TRAIN only, transform both)
#    This prevents data leakage from test set
# ============================================

# --- Encoding ---
# Label Encoding (fit on train, transform both)
_label_cols = [c for c in ['cut', 'color', 'clarity'] if c in X_train.columns and X_train[c].dtype == 'object']
for col in _label_cols:
    _le = LabelEncoder()
    X_train[col] = _le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].map({v: i for i, v in enumerate(_le.classes_)})
    X_test[col] = X_test[col].fillna(-1).astype(int)

# Auto-encode remaining categorical columns
for col in X_train.select_dtypes(include=['object', 'category']).columns:
    _le = LabelEncoder()
    X_train[col] = _le.fit_transform(X_train[col].astype(str))
    _mapping = {v: i for i, v in enumerate(_le.classes_)}
    X_test[col] = X_test[col].map(_mapping).fillna(-1).astype(int)

# Fill any remaining NaN (using train medians)
if X_train.isna().sum().sum() > 0:
    _medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(_medians)
    X_test = X_test.fillna(_medians)

# Align test columns to match training
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_train.columns]

# ============================================
# 6. ADVANCED PREPROCESSING
# ============================================

# Feature Scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Target Transform (Yeo-Johnson for skewed regression target)
pt = PowerTransformer(method='yeo-johnson')
y_train = pt.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# ============================================
# 7. MODEL TRAINING — ensemble
# ============================================
best_name = 'ensemble'
model = VotingRegressor(
    estimators=[
        ('random_forest', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('gradient_boosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),
        ('xgboost', XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),
    ]
)

# Cross-validation (same as pipeline)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f'Cross-validation scores: {cv_scores}')
print(f'CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Inverse transform predictions
y_pred = pt.inverse_transform(y_pred.reshape(-1, 1)).ravel()

# ============================================
# 8. EVALUATION
# ============================================
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
print(f'\nBest Model ({best_name})')
print(f'R² Score: {r2:.4f}')
print(f'RMSE:     {rmse:.4f}')
print(f'MAE:      {mae:.4f}')

# Save metrics
with open('stats/model_performance.txt', 'w') as f:
    f.write(f'Best Model (ensemble)\n')
    f.write(f'R2 Score: {r2:.4f}\n')
    f.write(f'MSE: {mse:.4f}\n')
    f.write(f'RMSE: {rmse:.4f}\n')
    f.write(f'MAE: {mae:.4f}\n')
    f.write(f'CV Mean: {cv_scores.mean():.4f}\n')

# ============================================
# 9. VISUALIZATION
# ============================================
try:
    # Correlation heatmap
    num_data = data.select_dtypes(include='number')
    plt.figure(figsize=(10, 8))
    sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=150)
    plt.close()

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=150)
    plt.close()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('plots/residual_plot.png', dpi=150)
    plt.close()

    print('\nPlots saved to plots/')
except Exception as e:
    print(f'Plotting error: {e}')

print('\n✓ Analysis complete!')
