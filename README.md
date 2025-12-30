# AutoEDA - Intelligent AutoML System

**Automated Exploratory Data Analysis and Machine Learning Pipeline**

A production-ready AutoML system that uses 7 specialized AI agents to analyze datasets, determine optimal strategies, train models, and generate production-ready Python code automatically.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The 7 Agents](#the-7-agents)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Workflow Details](#workflow-details)
- [Output](#output)
- [Advanced Usage](#advanced-usage)
- [Technical Details](#technical-details)

---

## 🎯 Overview

AutoEDA is an intelligent AutoML system that automates the entire machine learning pipeline from data analysis to code generation. It uses a multi-agent architecture where each agent specializes in a specific task, powered by AI (Google Gemini Flash) for strategic decisions and traditional ML libraries for computational tasks.

### Key Features

- ✅ **Fully Automated** - One command runs the entire pipeline
- ✅ **AI-Powered Strategy** - Intelligent decision-making for preprocessing, feature engineering, and optimization
- ✅ **Multi-Model Training** - Automatically trains 4 models (Logistic Regression, Random Forest, Decision Tree, XGBoost)
- ✅ **Iterative Optimization** - Retries with improvements if target accuracy not met
- ✅ **Code Generation** - Produces production-ready Python scripts
- ✅ **Zero Duplication** - Clean architecture with single source of truth
- ✅ **Flexible Deployment** - Run as unified service or separate microservices

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────┐
│               Orchestrator (main.py)                │
│         Coordinates all agents & workflow           │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│          Unified Service (unified_service.py)       │
│        Router - Imports from agents/ folder         │
│              Port 8000 - All endpoints              │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Agent 1    │  │   Agent 2    │  │   Agent 3    │
│   Analysis   │  │   Insight    │  │   Project    │
└──────────────┘  └──────────────┘  └──────────────┘
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Agent 4    │  │   Agent 5    │  │   Agent 6    │
│Preprocessing │  │   Feature    │  │    Model     │
└──────────────┘  └──────────────┘  └──────────────┘
                         ▼
                  ┌──────────────┐
                  │   Agent 7    │
                  │  Evaluation  │
                  └──────────────┘
```

### Design Principles

1. **Separation of Concerns** - Each agent has a specific responsibility
2. **Single Source of Truth** - All logic in `agents/` folder, no duplication
3. **AI + Computation** - AI for strategy, traditional ML for execution
4. **Microservices-Ready** - Can run as one service or seven separate services
5. **Clean Code** - No duplicate code, all imports from shared modules

---

## 🤖 The 7 Agents

### 1. **Analysis Agent** 📊
**Purpose:** Dataset analysis and type detection  
**Type:** Computational (no AI)  
**Location:** `agents/analysis_service/handler.py`

**What it does:**
- Detects data types (numerical, categorical, datetime)
- Identifies missing values
- Calculates cardinality for categorical columns
- Determines target column
- Returns comprehensive dataset summary

**Example Output:**
```json
{
  "columns": ["Age", "Gender", "Salary", "Exited"],
  "shape": [10000, 14],
  "data_types": {
    "numerical": ["Age", "Salary"],
    "categorical": ["Gender"],
    "datetime": []
  },
  "cardinality": {"Gender": 2},
  "target_column": "Exited"
}
```

---

### 2. **Insight Agent** 💡
**Purpose:** Generate meaningful insights from analysis  
**Type:** AI-powered (Google Gemini Flash)  
**Location:** `agents/insight_service/service.py`

**What it does:**
- Analyzes dataset characteristics
- Identifies patterns and anomalies
- Provides business insights
- Suggests areas of focus

**Example Output:**
```
- High cardinality in 'Surname' (2932 unique) - consider dropping
- Gender is balanced (54% Male, 46% Female)
- Age distribution is right-skewed
- Strong correlation between Geography and Exited
```

---

### 3. **Preprocessing Agent** 🧹
**Purpose:** Determine optimal preprocessing strategies  
**Type:** AI-powered (Google Gemini Flash)  
**Location:** `agents/preprocessing_service/service.py`

**What it does:**
- Chooses null handling methods (mean/median/mode/knn/drop)
- Selects outlier detection technique (IQR/Z-score/Isolation Forest)
- Determines scaling method (Standard/MinMax/Robust)

**Example Output:**
```json
{
  "null_strategy": {
    "Age": "median",
    "Salary": "mean"
  },
  "outlier_strategy": "iqr",
  "scaling_strategy": "standard"
}
```

---

### 4. **Feature Engineering Agent** 🔧
**Purpose:** Design feature transformations and encoding  
**Type:** AI-powered (Google Gemini Flash)  
**Location:** `agents/feature_service/service.py`

**What it does:**
- Recommends features to drop
- Suggests new features to create
- Chooses encoding methods per categorical column:
  - **One-Hot:** Low cardinality (< 10 unique)
  - **Label:** Ordinal categories
  - **Target:** High cardinality (> 20 unique)
  - **Ordinal:** Natural order exists
- Proposes numerical transformations (log, sqrt, polynomial)

**Example Output:**
```json
{
  "features_to_drop": ["RowNumber", "CustomerId"],
  "categorical_encoding": {
    "one_hot": ["Geography"],
    "label": ["Gender"]
  },
  "numerical_transformations": {
    "log_transform": ["Balance"]
  }
}
```

---

### 5. **Model Training Agent** 🤖
**Purpose:** Train and evaluate ML models  
**Type:** Computational (scikit-learn, XGBoost)  
**Location:** `agents/model_service/service.py`

**What it does:**
- Auto-detects problem type (classification/regression)
- Trains 4 models:
  - **Classification:** Logistic Regression, Random Forest, Decision Tree, XGBoost
  - **Regression:** Linear Regression, Random Forest, Decision Tree, XGBoost
- Applies preprocessing and feature engineering
- Returns performance metrics

**Example Output:**
```json
{
  "models": {
    "logistic_regression": {"score": 0.803, "metric": "accuracy"},
    "random_forest": {"score": 0.873, "metric": "accuracy"},
    "decision_tree": {"score": 0.790, "metric": "accuracy"},
    "xgboost": {"score": 0.867, "metric": "accuracy"}
  },
  "best_model": "random_forest",
  "best_score": 0.873
}
```

---

### 6. **Evaluation Agent** 📈
**Purpose:** Evaluate performance and suggest improvements  
**Type:** AI-powered (Google Gemini Flash)  
**Location:** `agents/evaluation_service/service.py`

**What it does:**
- Compares best score against target accuracy (default 85%)
- Decides if retry is needed
- Suggests specific improvements:
  - Feature engineering adjustments
  - Hyperparameter tuning
  - Data quality improvements

**Example Output:**
```json
{
  "meets_target": true,
  "best_model": "random_forest",
  "accuracy": 0.873,
  "suggestions": [],
  "retry_needed": false
}
```

---

### 7. **Project Agent** 📁
**Purpose:** Generate production-ready code  
**Type:** AI-powered (Google Gemini Flash)  
**Location:** `agents/project_service/handler.py`

**What it does:**
- Generates complete `analysis.py` script with:
  - Data loading
  - Exploratory Data Analysis (EDA)
  - Categorical encoding
  - Model training
  - Results saving
- Creates `README.md` with documentation
- Ensures code is clean (removes markdown, explanatory text)

**Example Output:**
Complete Python script ready to run!

---

## 🔄 How It Works

### Workflow Overview

```
1. Load Dataset → 2. Analysis → 3. Insights → 4. Preprocessing Strategy
                                                           ↓
6. Project Generation ← 5. Evaluation ← 4. Model Training ← Feature Strategy
```

### Detailed Steps

1. **User runs:** `cd orchestrator && python main.py`
2. **Orchestrator starts unified service** on port 8000
3. **Analysis Agent** analyzes the dataset
4. **Insight Agent** generates insights
5. **Preprocessing Agent** determines strategies
6. **Feature Engineering Agent** recommends transformations
7. **Model Training Agent** trains 4 models
8. **Evaluation Agent** checks if target met:
   - ✅ If yes → proceed to step 9
   - ❌ If no → retry from step 5 (up to 3 iterations)
9. **Project Agent** generates production code
10. **Human approval** for file creation
11. **Files written** to `autoeda_output_<timestamp>/`
12. **Service stopped** automatically

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd autoeda
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create `.env` file:**
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_ai_key_here
```

Get your API key from: https://aistudio.google.com/app/apikey

4. **Add your dataset:**
Place your CSV file in `orchestrator/` and update the filename in `orchestrator/main.py` (line 18)

---

## 🚀 Quick Start

### Run the System

```bash
cd orchestrator
python main.py
```

That's it! The system will:
- ✅ Start the unified service
- ✅ Run the complete AutoML pipeline
- ✅ Generate production-ready code
- ✅ Stop the service

### Expected Output

```
🚀 Starting AutoEDA Unified Service...
✅ Service is ready!

📊 Step 1/6: Analyzing dataset...
  → Sending to http://localhost:8000/a2a/analysis
  ✓ Response received
✅ Analysis complete!

💡 Step 2/6: Generating insights...
✅ Insights generated!

🧹 Step 3/6: Determining preprocessing strategy...
✅ Preprocessing strategy determined!

🔧 Step 4/6: Designing feature engineering...
✅ Feature engineering designed!

🤖 Step 5/6: Training models...
  [MODEL] Training 4 models...
  [MODEL] Best model: random_forest (0.8730)
✅ Models trained!

📈 Step 6/6: Evaluating performance...
✅ Target accuracy met! (87.30% >= 85.00%)

📁 Generating project files...
✅ Project generation complete!

🎉 AutoML pipeline completed successfully!
📂 Output: autoeda_output_1767082323
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Service Configuration
SERVICE_PORT = 8000              # Port for unified service
SERVICE_HOST = "localhost"       # Host address
SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"

# AutoML Parameters
MAX_OPTIMIZATION_ITERATIONS = 3  # Max retry attempts
TARGET_ACCURACY = 0.85           # Target accuracy (85%)
```

---

## 📁 Project Structure

```
autoeda/
├── agents/                      # All agent services
│   ├── analysis_service/
│   │   ├── handler.py          # Core analysis logic
│   │   └── service.py          # FastAPI wrapper
│   ├── insight_service/
│   │   └── service.py          # AI insight generation
│   ├── project_service/
│   │   ├── handler.py          # Core project logic
│   │   └── service.py          # FastAPI wrapper
│   ├── preprocessing_service/
│   │   └── service.py          # AI preprocessing strategy
│   ├── feature_service/
│   │   └── service.py          # AI feature engineering
│   ├── model_service/
│   │   └── service.py          # Model training
│   └── evaluation_service/
│       └── service.py          # AI evaluation
│
├── orchestrator/
│   ├── main.py                 # Main workflow orchestrator
│   └── churn.csv               # Your dataset
│
├── a2a/                        # Agent-to-Agent communication
│   ├── schemas.py              # Request/Response models
│   └── client.py               # HTTP client
│
├── core/
│   ├── hitl.py                 # Human-in-the-loop
│   └── file_writer.py          # File generation
│
├── unified_service.py          # Single entry point (router)
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── .env                        # API keys
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

---

## 🔄 Workflow Details

### Iteration Loop

If the target accuracy is not met, the system automatically retries:

```
Iteration 1: Train → Evaluate → ❌ (82% < 85%)
             ↓
Iteration 2: Improve Features → Train → Evaluate → ❌ (84% < 85%)
             ↓
Iteration 3: Further Improve → Train → Evaluate → ✅ (87% >= 85%)
             ↓
             Generate Code
```

### Human-in-the-Loop

Before writing files, the system asks for approval:

```
📁 Ready to create project files:
  - analysis.py
  - data.csv
  - README.md

Proceed? (yes/no): yes
✅ Files created successfully!
```

---

## 📦 Output

### Generated Files

```
autoeda_output_<timestamp>/
├── stats/
│   └── model_performance.txt   # Model accuracies
├── plots/
│   ├── histogram_*.png         # Numerical distributions
│   └── bar_chart_*.png         # Categorical distributions
├── reports/
│   └── classification_report.txt
├── analysis.py                 # Production-ready script
├── data.csv                    # Dataset copy
└── README.md                   # Documentation
```

### Running Generated Code

```bash
cd autoeda_output_<timestamp>
python analysis.py
```

The script will:
- Load data
- Perform EDA
- Encode categorical features
- Train 4 models
- Save results to stats/, plots/, reports/

---

## 🔧 Advanced Usage

### Option 1: Unified Service (Default)

```bash
cd orchestrator && python main.py
```
- Single port (8000)
- All agents accessible
- Recommended for production

### Option 2: Separate Services

```bash
python launch.py
```
- 7 separate ports (8001-8007)
- Good for debugging individual agents
- Microservices architecture

### Manual Service Control

Start service manually:
```bash
uvicorn unified_service:app --port 8000
```

Test individual endpoint:
```bash
curl -X POST http://localhost:8000/a2a/analysis \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test", "sender": "user", "input": {"csv_path": "data.csv"}}'
```

---

## 🔬 Technical Details

### Technologies Used

- **FastAPI** - Web framework for agents
- **CrewAI** - AI agent orchestration
- **Gemini** - LLM API (Flash Latest)
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization

### AI Model

- **Provider:** Google
- **Model:** Gemini Flash Latest
- **Purpose:** Strategic decision-making
- **Usage:** 5 out of 7 agents use AI

### Architecture Benefits

1. **Zero Code Duplication** - `unified_service.py` imports from `agents/`
2. **Single Source of Truth** - All logic in agent files
3. **Easy Maintenance** - Fix once, works everywhere
4. **Flexible Deployment** - One service or many
5. **Clean Separation** - AI vs Computational agents

---

## 📝 Dependencies

```
fastapi
uvicorn
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
crewai
langchain-google-genai
python-dotenv
requests
```

---

## 🎯 Example Use Case

**Scenario:** Customer churn prediction

1. Place `churn.csv` in `orchestrator/`
2. Run `cd orchestrator && python main.py`
3. System analyzes 10,000 customers with 14 features
4. Trains 4 models, achieves 87.3% accuracy
5. Generates production-ready `analysis.py`
6. You get complete working code in 2 minutes!

---

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgments

Built with:
- CrewAI for agent orchestration
- Google Gemini for fast & smart LLM inference
- scikit-learn & XGBoost for ML

---

**Made with ❤️ by the AutoEDA Team**

For questions or support, please open an issue on GitHub.
