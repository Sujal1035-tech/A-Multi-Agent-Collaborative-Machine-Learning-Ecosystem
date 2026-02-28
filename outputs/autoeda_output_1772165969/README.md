# AutoEDA Analysis Results

## Best Model: lightgbm_tuned
## Score: 0.9052
## Problem Type: regression

### How to Run
```bash
pip install -r requirements.txt
python analysis.py
```

### Pipeline Decisions (Reproduced in analysis.py)

#### Preprocessing
- Null Handling: median
- Outlier Capping: IQR (threshold=1.5) on 2 columns

#### Feature Encoding
- One-Hot: ['cylinders', 'origin']
- Frequency Encoding: ['name']

#### Advanced Processing
- Train/Test Split: 80/20 
- Scaling: StandardScaler
- Class Balancing: Model-level class weighting (`class_weight='balanced'` where supported), no SMOTE
- Target Transform: None

#### Model
- **lightgbm_tuned**
- Parameters: `{'n_estimators': 202, 'learning_rate': 0.05269284907890199, 'max_depth': 7, 'subsample': 0.7740385332157751, 'colsample_bytree': 0.922980191220747, 'random_state': 42, 'verbose': -1}`

### Output Files
- `models/` — Saved model and scaling artifacts (.pkl)
- `stats/model_performance.txt` — Score summary
- `reports/metrics.txt` — Detailed classification/regression report
- `plots/correlation_heatmap.png` — Feature correlations
- `plots/actual_vs_predicted.png` — Actual vs Predicted scatter
- `plots/residual_plot.png` — Residual analysis
