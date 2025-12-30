# Autonomous Multi-Agent Collaborative Machine Learning Ecosystem

## System Overview

This system completely automates the creation of machine learning models. Instead of relying on a human to manually clean data and select algorithms, it deploys a team of **seven specialized AI agents** that work together to solve the problem.

The process is simple: you provide the data, and the agents handle the rest. They automatically analyze the dataset, fix quality issues, engineer features, and train multiple models to find the best performer. The final result is not just a prediction, but a complete, high-quality Python codebase.

**Proven Performance:**
The system has been tested on **10 datasets**, consistently achieving an accuracy of **more than 80%**. It automatically generates a comprehensive project folder (e.g., `autoeda_output_1767089156`) containing the full source code and analysis reports.

## Architectural Design

The system follows a Microservices-ready Hub-and-Spoke architecture, facilitated by a centralized router (`unified_service.py`).

### Communication Protocol
- **Transport**: HTTP/1.1 (REST)
- **Format**: JSON (Strict Schema Validation via Pydantic)
- **Routing**: Centralized unification on port 8081 (default), extensible to distributed ports.

### Agent Specifications

1.  **Analysis Agent** (`agents/analysis_service`)
    -   **Role**: Statistical Profiling & Type Inference
    -   **Mechanism**: Deterministic Pandas/NumPy analysis. Calculates distribution skew, cardinality, and nullity matrices.

2.  **Insight Agent** (`agents/insight_service`)
    -   **Role**: Semantic Pattern Recognition
    -   **Mechanism**: LLM-driven. Synthesizes statistical profiles into actionable business and data science insights.

3.  **Preprocessing Agent** (`agents/preprocessing_service`)
    -   **Role**: Cleaning Strategy Formulation
    -   **Mechanism**: LLM-driven. Determines optimal imputation strategies (KNN, Iterative), outlier mitigation (IQR, Z-Score), and scaling techniques based on data distribution.

4.  **Feature Engineering Agent** (`agents/feature_service`)
    -   **Role**: Feature Space Optimization
    -   **Mechanism**: LLM-driven. Designs encoding strategies (Target, One-Hot, Ordinal) and transformation pipelines (Log, Polynomial, Interactions) to maximize information gain.

5.  **Model Training Agent** (`agents/model_service`)
    -   **Role**: Model Training & Hyperparameter Tuning
    -   **Mechanism**: Computational (Scikit-Learn/XGBoost). Implements 5-fold Cross-Validation, SHAP value calculation, SMOTE for class imbalance, and Voting Classifier ensembling.

6.  **Evaluation Agent** (`agents/evaluation_service`)
    -   **Role**: Performance Validation
    -   **Mechanism**: LLM-driven. Analyzes confusion matrices and classification reports to determine if the model meets deployment criteria or requires iterative refinement.

7.  **Project Agent** (`agents/project_service`)
    -   **Role**: Code Synthesis
    -   **Mechanism**: LLM-driven. Generates PEP-8 compliant, executable Python code wrapping the entire discovered pipeline.

## Technical Stack

-   **Orchestration Logic**: Custom Python State Machine / CrewAI
-   **LLM Inference Engine**: Groq API (Model: `llama-3.3-70b-versatile`)
-   **API Framework**: FastAPI / Uvicorn
-   **Data Processing**: Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn
-   **Optimization**: Optuna (Bayesian Hyperparameter Optimization)
-   **Explainability**: SHAP (Shapley Additive Explanations)
-   **Resilience**: Custom exponential backoff retry logic for API rate limit handling.

## Installation and Configuration

### Prerequisites
-   Python 3.9+
-   Valid Groq API Key

### Deployment

1.  **Clone Repository**
    ```bash
    git clone https://github.com/Sujal1035-tech/A-Multi-Agent-Collaborative-Machine-Learning-Ecosystem.git
    cd autoeda
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **System Configuration**
    Modify `config.py` to adjust system parameters:
    -   `SERVICE_PORT`: API binding port (Default: 8081)
    -   `LLM_MODEL`: Target LLM model identifier
    -   `MAX_OPTIMIZATION_ITERATIONS`: Limit for iterative refinement loops

## Execution Manual

### Service Initialization
Start the unified API server to handle agent requests.
```bash
python -m uvicorn unified_service:app --port 8081 --reload
```

### Workflow Initiation
Execute the orchestrator to begin the AutoML pipeline.
```bash
cd orchestrator
python main.py
```

## Output Artifacts

Upon successful execution, the system generates a timestamped directory containing:
-   `analysis.py`: Complete, reproducible training pipeline source code.
-   `reports/`: Detailed classification reports and metric logs.
-   `plots/`: Visualization assets (Feature Importance, SHAP summaries).
-   `stats/`: Raw performance data in JSON format.
