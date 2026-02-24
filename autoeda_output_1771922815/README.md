# AutoEDA Analysis Results

## Best Model: ensemble
## Score: 0.9062
## Problem Type: regression

### How to Run
```bash
pip install -r requirements.txt
python analysis.py
```

### Pipeline Decisions (Reproduced in analysis.py)

#### Preprocessing
- Null Handling: knn, mode, median, mean
- Outlier Capping: IQR (threshold=1.5) on 2 columns

#### Feature Encoding
- One-Hot: ['low_cardinality_categorical_cols', 'origin']
- Label Encoding: ['binary_or_ordinal_categorical_cols', 'model_year']
- Frequency Encoding: ['high_cardinality_categorical_cols', 'name']

#### Advanced Processing
- Train/Test Split: 80/20 
- Scaling: StandardScaler
- Class Balancing: None
- Target Transform: None

#### Model
- **ensemble**
- Parameters: `{'ensemble_voting': 'average', 'random_state': 42}`

### Output Files
- `stats/model_performance.txt` — Score summary
- `reports/metrics.txt` — Detailed classification/regression report
- `plots/correlation_heatmap.png` — Feature correlations
- `plots/actual_vs_predicted.png` — Actual vs Predicted scatter
- `plots/residual_plot.png` — Residual analysis
