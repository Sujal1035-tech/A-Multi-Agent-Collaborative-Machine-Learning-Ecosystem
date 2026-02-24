# AutoEDA Analysis Results

## Best Model: random_forest_tuned
## Score: 0.9714
## Problem Type: classification

### How to Run
```bash
pip install -r requirements.txt
python analysis.py
```

### Pipeline Decisions (Reproduced in analysis.py)

#### Preprocessing
- Null Handling: mode, knn
- Outlier Capping: IQR (threshold=1.5) on 6 columns

#### Feature Encoding
- Auto label encoding for categorical columns

#### Advanced Processing
- Train/Test Split: 80/20 (Stratified)
- Scaling: StandardScaler
- Class Balancing: None
- Target Transform: None

#### Model
- **random_forest_tuned**
- Parameters: `{'n_estimators': 149, 'max_depth': 21, 'min_samples_split': 8, 'min_samples_leaf': 2, 'random_state': 42}`

### Output Files
- `stats/model_performance.txt` — Score summary
- `reports/metrics.txt` — Detailed classification/regression report
- `plots/correlation_heatmap.png` — Feature correlations
- `plots/confusion_matrix.png` — Confusion matrix

