# AutoEDA Analysis Results

## Best Model: ensemble
## Score: 0.9829
## Problem Type: regression

### How to Run
```bash
pip install -r requirements.txt
python analysis.py
```

### Pipeline Decisions (Reproduced in analysis.py)

#### Preprocessing
- No special preprocessing required

#### Feature Encoding
- Label Encoding: ['cut', 'color', 'clarity']

#### Advanced Processing
- Train/Test Split: 80/20 
- Scaling: StandardScaler
- Class Balancing: None
- Target Transform: Yeo-Johnson PowerTransformer

#### Model
- **ensemble**
- Parameters: `{'ensemble_voting': 'average', 'random_state': 42}`

### Output Files
- `stats/model_performance.txt` — Score summary
- `reports/metrics.txt` — Detailed classification/regression report
- `plots/correlation_heatmap.png` — Feature correlations
- `plots/actual_vs_predicted.png` — Actual vs Predicted scatter
- `plots/residual_plot.png` — Residual analysis
