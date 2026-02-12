# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
# Create directories
os.makedirs('stats', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)
# Load data
try:
    data = pd.read_csv('data.csv')
    print("Data loaded successfully.")
except Exception as e:
    print("Error loading data: ", str(e))
# Perform EDA
numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(10,8))
sns.heatmap(numeric_data.corr(), annot=True)
plt.savefig('plots/correlation_heatmap.png')
plt.close()
plt.figure(figsize=(10,8))
sns.boxplot(data=numeric_data)
plt.savefig('plots/box_plots.png')
plt.close()
numeric_data.hist(figsize=(10,10))
plt.savefig('plots/histograms.png')
plt.close()
# Encode categorical columns
le = LabelEncoder()
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
# Split data into training and testing sets
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': xgb.XGBClassifier()
}
model_accuracies = {}
for model_name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy
        print(f"Model: {model_name}, Accuracy: {accuracy}")
        # Save model performance
        with open('stats/model_performance.txt', 'a') as f:
            f.write(f"Model: {model_name}, Accuracy: {accuracy}\n")
        # Save detailed reports
        with open(f'reports/{model_name}_report.txt', 'w') as f:
            f.write(classification_report(y_test, y_pred))
        # Save confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
        plt.title(f'Confusion Matrix {model_name}')
        plt.savefig(f'plots/{model_name}_confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
# Find the best model
best_model = max(model_accuracies, key=model_accuracies.get)
print(f"Best Model: {best_model}")
# Save best model performance
with open('stats/model_performance.txt', 'a') as f:
    f.write(f"Best Model: {best_model}\n")
# Save summary
with open('stats/summary.txt', 'w') as f:
    f.write(f"Best Model: {best_model}\n")
    f.write(f"Best Model Accuracy: {model_accuracies[best_model]}\n")