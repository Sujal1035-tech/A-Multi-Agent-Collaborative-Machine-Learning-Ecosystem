"""
AutoEDA Generated Analysis
This code reproduces the exact pipeline that the agents performed.
Best Model: random_forest_tuned
Problem Type: classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Model Import
from sklearn.ensemble import RandomForestClassifier

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
TARGET_COL = 'Class'
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
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratified for classification
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# ============================================
# 5. PREPROCESSING (fit on TRAIN only, transform both)
#    This prevents data leakage from test set
# ============================================

# --- Null handling ---
if 'Bare.nuclei' in X_train.columns and X_train['Bare.nuclei'].dtype in ['int64', 'float64']:
    try:
        from sklearn.impute import KNNImputer
        _imputer = KNNImputer(n_neighbors=5)
        X_train[['Bare.nuclei']] = _imputer.fit_transform(X_train[['Bare.nuclei']])
        X_test[['Bare.nuclei']] = _imputer.transform(X_test[['Bare.nuclei']])
    except ImportError:
        _fill = X_train['Bare.nuclei'].median()
        X_train['Bare.nuclei'] = X_train['Bare.nuclei'].fillna(_fill)
        X_test['Bare.nuclei'] = X_test['Bare.nuclei'].fillna(_fill)
if 'Bl.cromatin' in X_train.columns and X_train['Bl.cromatin'].dtype in ['int64', 'float64']:
    try:
        from sklearn.impute import KNNImputer
        _imputer = KNNImputer(n_neighbors=5)
        X_train[['Bl.cromatin']] = _imputer.fit_transform(X_train[['Bl.cromatin']])
        X_test[['Bl.cromatin']] = _imputer.transform(X_test[['Bl.cromatin']])
    except ImportError:
        _fill = X_train['Bl.cromatin'].median()
        X_train['Bl.cromatin'] = X_train['Bl.cromatin'].fillna(_fill)
        X_test['Bl.cromatin'] = X_test['Bl.cromatin'].fillna(_fill)
if 'Normal.nucleoli' in X_train.columns and X_train['Normal.nucleoli'].dtype in ['int64', 'float64']:
    try:
        from sklearn.impute import KNNImputer
        _imputer = KNNImputer(n_neighbors=5)
        X_train[['Normal.nucleoli']] = _imputer.fit_transform(X_train[['Normal.nucleoli']])
        X_test[['Normal.nucleoli']] = _imputer.transform(X_test[['Normal.nucleoli']])
    except ImportError:
        _fill = X_train['Normal.nucleoli'].median()
        X_train['Normal.nucleoli'] = X_train['Normal.nucleoli'].fillna(_fill)
        X_test['Normal.nucleoli'] = X_test['Normal.nucleoli'].fillna(_fill)
if 'Mitoses' in X_train.columns and X_train['Mitoses'].dtype in ['int64', 'float64']:
    try:
        from sklearn.impute import KNNImputer
        _imputer = KNNImputer(n_neighbors=5)
        X_train[['Mitoses']] = _imputer.fit_transform(X_train[['Mitoses']])
        X_test[['Mitoses']] = _imputer.transform(X_test[['Mitoses']])
    except ImportError:
        _fill = X_train['Mitoses'].median()
        X_train['Mitoses'] = X_train['Mitoses'].fillna(_fill)
        X_test['Mitoses'] = X_test['Mitoses'].fillna(_fill)
if 'Class' in X_train.columns:
    _fill = X_train['Class'].mode()[0] if len(X_train['Class'].mode()) > 0 else 'Unknown'
    X_train['Class'] = X_train['Class'].fillna(_fill)
    X_test['Class'] = X_test['Class'].fillna(_fill)

# --- Outlier handling (IQR bounds from train only) ---
for col in ['Marg.adhesion', 'Epith.c.size', 'Bare.nuclei', 'Bl.cromatin', 'Normal.nucleoli', 'Mitoses']:
    if col in X_train.columns and X_train[col].dtype in ['int64', 'float64']:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)  # Same bounds from train

# --- Encoding ---
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

# ============================================
# 7. MODEL TRAINING — random_forest_tuned
# ============================================
best_name = 'random_forest_tuned'
model = RandomForestClassifier(n_estimators=149, max_depth=21, min_samples_split=8, min_samples_leaf=2, random_state=42)

# Cross-validation (same as pipeline)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================
# 8. EVALUATION
# ============================================
acc = accuracy_score(y_test, y_pred)
print(f'\nBest Model ({best_name}) Accuracy: {acc:.4f}')

# Detailed metrics
avg = 'binary' if len(y.unique()) == 2 else 'weighted'
prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
print(f'Precision: {prec:.4f}')
print(f'Recall:    {rec:.4f}')
print(f'F1 Score:  {f1:.4f}')

# Classification Report
report = classification_report(y_test, y_pred)
print('\nClassification Report:\n', report)

# Save metrics
with open('stats/model_performance.txt', 'w') as f:
    f.write(f'Best Model (random_forest_tuned): {acc:.4f}\n')
    f.write(f'Precision: {prec:.4f}\n')
    f.write(f'Recall: {rec:.4f}\n')
    f.write(f'F1 Score: {f1:.4f}\n')
    f.write(f'CV Mean: {cv_scores.mean():.4f}\n')
with open('reports/metrics.txt', 'w') as f:
    f.write('Classification Report\n')
    f.write('=' * 50 + '\n\n')
    f.write(report)

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

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.close()

    print('\nPlots saved to plots/')
except Exception as e:
    print(f'Plotting error: {e}')

print('\n✓ Analysis complete!')
