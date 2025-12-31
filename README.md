# Autonomous Multi-Agent Collaborative Machine Learning Ecosystem

## System Overview

This system completely automates the creation of machine learning models using **seven specialized AI agents**. The LLM intelligently  feature engineering based on data characterdecides preprocessing strategies, encoding techniques, andistics.

**Key Features:**
- 🎯 **Interactive Input**: Enter CSV path (local or URL) and select target column
- 🧠 **Smart Preprocessing**: LLM decides null handling (mean/median/mode/KNN) per column
- 🔧 **Smart Encoding**: LLM decides encoding (one-hot/label/frequency) based on cardinality
- 📊 **Auto Model Training**: Cross-validation, SMOTE, Optuna tuning, ensemble methods
- 📝 **Code Generation**: Produces deployable Python code

**Proven Performance:** Tested on multiple datasets, achieving **>80% accuracy** consistently.

---

## Architecture

```
User Input (CSV + Target Column)
        ↓
┌─────────────────┐
│ Analysis Agent  │ → Stats, outliers, skewness, null %
└────────┬────────┘
         ↓
┌─────────────────────┐
│ Preprocessing Agent │ → LLM decides + executes null/outlier handling
└────────┬────────────┘
         ↓
┌─────────────────┐
│  Feature Agent  │ → LLM decides + executes smart encoding
└────────┬────────┘
         ↓
┌─────────────────┐
│  Model Agent    │ → Train, tune, ensemble, SHAP analysis
└────────┬────────┘
         ↓
┌─────────────────┐
│  Project Agent  │ → Generate deployable Python code
└─────────────────┘
```

---

## Agent Specifications

| Agent | Role | Mechanism |
|-------|------|-----------|
| **Analysis** | Statistical profiling | Python: outlier detection (IQR), skewness, null %, cardinality |
| **Insight** | Business insights | LLM-driven pattern recognition |
| **Preprocessing** | Data cleaning | LLM decides + executes: KNN/median/mode imputation, IQR capping, scaling |
| **Feature** | Smart encoding | LLM decides + executes: one-hot, label, frequency encoding |
| **Model** | Training & tuning | Scikit-Learn, XGBoost, Optuna, SMOTE, 5-fold CV, SHAP |
| **Evaluation** | Performance review | LLM analyzes confusion matrix, suggests improvements |
| **Project** | Code generation | LLM generates PEP-8 compliant Python code |

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Orchestration | Custom Python State Machine / CrewAI |
| LLM | Google Gemini (`gemini-flash`) |
| API | FastAPI / Uvicorn |
| ML | Pandas, NumPy, Scikit-Learn, XGBoost |
| Optimization | Optuna (Bayesian tuning) |
| Explainability | SHAP |
| Imbalance | SMOTE (imbalanced-learn) |

---

## Quick Start

### 1. Install
```bash
git clone https://github.com/Sujal1035-tech/A-Multi-Agent-Collaborative-Machine-Learning-Ecosystem.git
cd autoeda
pip install -r requirements.txt
```

### 2. Configure
Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run
```bash
# Start service
python -m uvicorn unified_service:app --port 8081 --reload

# In another terminal - run workflow
cd orchestrator
python main.py
```

### 4. Interactive Input
```
📂 Enter path to CSV file (local path or URL):
> https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv

📋 Available columns:
   1. sepal_length
   2. sepal_width
   3. petal_length
   4. petal_width
   5. species

🎯 Enter target column name (or number):
> species
```

---

## Smart Preprocessing (LLM-Driven)

The system uses LLM to decide optimal strategies per column:

| Data Characteristic | LLM Decision |
|---------------------|--------------|
| Null + Skewed data | Median imputation |
| Null + Normal data | Mean imputation |
| Null + Categorical | Mode imputation |
| >5% Outliers | IQR capping |
| Cardinality 2-5 | One-hot encoding |
| Cardinality 6-10 | Label encoding |
| Cardinality >10 | Frequency encoding |

---

## Output Artifacts

Generated project folder contains:
- `data.csv` - Copy of input dataset
- `analysis.py` - Complete training pipeline
- `README.md` - Project documentation
- `reports/` - Classification reports per model
- `plots/` - Confusion matrices, correlation heatmaps
- `stats/` - Performance metrics

---

## Test Datasets

| Dataset | Type | URL | Target |
|---------|------|-----|--------|
| Iris | Classification | `https://...seaborn.../iris.csv` | species |
| Titanic | Classification | `https://...datasets.../titanic.csv` | Survived |
| Tips | Regression | `https://...seaborn.../tips.csv` | tip |

---

## License

MIT License
