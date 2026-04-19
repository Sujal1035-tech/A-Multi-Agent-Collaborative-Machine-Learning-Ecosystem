# AutoEDA Analysis Results

## Best Model: Unknown
## Score: 0.0000
## Problem Type: regression

### How to Run
```bash
pip install -r requirements.txt
python analysis.py
```

### Pipeline Decisions (Reproduced in analysis.py)

#### Preprocessing
- Dropped Features: ['student_id']

#### Feature Encoding
- One-Hot: ['gender', 'academic_level', 'internet_quality']

#### Advanced Processing
- Train/Test Split: 80/20 
- Scaling: StandardScaler
- Class Balancing: Model-level class weighting (`class_weight='balanced'` where supported), no SMOTE
- Target Transform: None

#### Model
- **Unknown**
- Parameters: `{'random_state': 42}`

### Output Files
- `models/` — Saved model and scaling artifacts (.pkl)
- `stats/model_performance.txt` — Score summary
- `reports/metrics.txt` — Detailed classification/regression report
- `plots/correlation_heatmap.png` — Feature correlations
- `plots/actual_vs_predicted.png` — Actual vs Predicted scatter
- `plots/residual_plot.png` — Residual analysis
