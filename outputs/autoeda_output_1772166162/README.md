# AutoEDA Analysis Results

## Best Model: random_forest_tuned
## Score: 0.9643
## Problem Type: classification

### How to Run
```bash
pip install -r requirements.txt
python analysis.py
```

### Pipeline Decisions (Reproduced in analysis.py)

#### Preprocessing
- Null Handling: median
- Outlier Capping: IQR (threshold=1.5) on 6 columns

#### Feature Encoding
- Auto label encoding for categorical columns

#### Advanced Processing
- Train/Test Split: 80/20 (Stratified)
- Scaling: StandardScaler
- Class Balancing: Model-level class weighting (`class_weight='balanced'` where supported), no SMOTE
- Target Transform: None

#### Model
- **random_forest_tuned**
- Parameters: `{'n_estimators': 427, 'max_depth': 17, 'min_samples_split': 8, 'min_samples_leaf': 1, 'random_state': 42, 'class_weight': 'balanced'}`

### Output Files
- `models/` — Saved model and scaling artifacts (.pkl)
- `stats/model_performance.txt` — Score summary
- `reports/metrics.txt` — Detailed classification/regression report
- `plots/correlation_heatmap.png` — Feature correlations
- `plots/confusion_matrix.png` — Confusion matrix

