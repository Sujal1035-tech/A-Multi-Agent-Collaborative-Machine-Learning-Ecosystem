# AutoML - Multi-Agent Machine Learning System

## What is This?

Ever spent hours cleaning data, trying different encodings, and tuning models? This project does all of that automatically.

You give it a dataset. Seven AI agents work together to:
1. Understand your data (what's missing, what's weird, what matters)
2. Clean it up (the smart way - not just "fill with mean everywhere")
3. Encode categories properly (one-hot for small groups, frequency for large)
4. Train multiple models and pick the best one
5. Generate ready-to-use Python code

The cool part? **The AI actually thinks about your data**. High skewness? It uses median. Many unique values? It uses frequency encoding. The decisions are context-aware, not hardcoded.

---

## How It Works

```
Your CSV (local file or URL)
         ↓
   Pick target column
         ↓
┌────────────────────┐
│  Analysis Agent    │  Examines: nulls, outliers, distributions
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Preprocessing Agent│  LLM decides: how to handle each column's issues
└─────────┬──────────┘
          ↓
┌────────────────────┐
│   Feature Agent    │  LLM decides: best encoding per column
└─────────┬──────────┘
          ↓
┌────────────────────┐
│    Model Agent     │  Trains: LogReg, RF, XGBoost, ensembles
└─────────┬──────────┘
          ↓
   Deployable Python code
```

---

## The Seven Agents

| Agent | What it Does |
|-------|--------------|
| **Analysis** | Counts nulls, detects outliers (IQR), measures skewness, checks cardinality |
| **Insight** | Generates human-readable observations about your data |
| **Preprocessing** | LLM picks: median for skewed, mean for normal, mode for categories |
| **Feature** | LLM picks: one-hot (2-5 values), label (ordinal), frequency (10+ values) |
| **Model** | Cross-validates, balances with SMOTE, tunes with Optuna, creates ensembles |
| **Evaluation** | Reviews confusion matrices, suggests improvements |
| **Project** | Writes clean Python code you can actually use |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Sujal1035-tech/A-Multi-Agent-Collaborative-Machine-Learning-Ecosystem.git
cd autoeda
pip install -r requirements.txt

# Add your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run
python -m uvicorn unified_service:app --port 8081 --reload
# In another terminal:
cd orchestrator && python main.py
```

You'll see prompts like:
```
📂 Enter path to CSV file:
> https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv

🎯 Enter target column:
> species
```

---

## What Makes It Smart?

The LLM doesn't just follow rules. It looks at your actual data:

| It Sees | It Does |
|---------|---------|
| Column with 20% nulls + high skew | Fills with median |
| Column with 3 unique values | One-hot encodes |
| Column with 500 unique values | Frequency encodes |
| 15% outliers in a feature | Caps using IQR bounds |
| Class imbalance | Applies SMOTE |
| Accuracy below 80% | Triggers Optuna tuning |


## Output

After running, you get a folder with:
- `data.csv` - Your dataset
- `analysis.py` - Full training pipeline
- `reports/` - Classification reports, insights
- `plots/` - Confusion matrices, heatmaps

## Demo Outputs

The repository includes sample output folders for reference:

| Folder | Dataset | Task | Best Model | Rows | Columns | Compute Time |
|--------|---------|------|------------|------|---------|--------------|
| `autoeda_output_1771922815/` | Auto MPG | Regression | Ensemble (Score: 0.9062) | 398 | 9 | 3m 44s |
| `autoeda_output_1771923341/` | Breast Cancer | Classification | Random Forest Tuned (Score: 0.9714) | 699 | 11 | 3m 20s |
| `autoeda_output_1771930503/` | Diamonds | Regression | Ensemble (Score: 0.9829) | 53,940 | 10 | m 19.51s |

These demonstrate the system's capabilities on both regression and classification problems.


## License

MIT - Use it however you want.
