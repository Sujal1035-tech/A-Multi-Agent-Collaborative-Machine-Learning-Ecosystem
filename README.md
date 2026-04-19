# AutoML: The Multi-Agent Machine Learning Ecosystem

> **Your intelligent AI data scientist that thinks, cleans, trains, and writes code for you.**

Ever spent hours meticulously cleaning data, experimenting with different encodings, and agonizing over model tuning? **AutoML** does all of that automatically.

Give it a dataset and a target column. Seven specialized AI agents will work collaboratively to understand your data, clean it intelligently, train multiple models, find the best fit, and give you production-ready Python code.

The best part? **The AI actually thinks about your data contextually.**
High skewness? It defaults to median imputation. Dozens of unique values? It uses frequency encoding over one-hot encoding. All decisions are context-aware, not just a hardcoded list of rules.

---

## Meet Your AI Data Team

Here are the 7 AI agents working behind the scenes for you:

1. **Analysis Agent:** Counts nulls, detects outliers (IQR), measures skewness, and evaluates cardinality.
2. **Insight Agent:** Translates raw statistics into human-readable, meaningful observations about your dataset.
3. **Preprocessing Agent:** An LLM that decides the smartest way to handle data gaps (e.g., median for skewed data, mode for categoricals).
4. **Feature Agent:** Determines the optimal encoding technique per column (One-Hot for 2-5 values, Label for ordinal, Frequency for high cardinality).
5. **Model Agent:** Cross-validates, applies SMOTE for imbalance, aggressively tunes using Optuna, and creates powerful ensembles.
6. **Evaluation Agent:** Reviews model performance with confusion matrices and suggests tangible improvements.
7. **Project Agent:** Writes a clean, fully documented, and ready-to-deploy Python script based on everything the team learned.

---

## Quick Start

Get your new AI data team up and running in minutes!

### 1. Installation

```bash
git clone https://github.com/Sujal1035-tech/A-Multi-Agent-Collaborative-Machine-Learning-Ecosystem.git
cd autoeda
pip install -r requirements.txt
```

### 2. Configuration
You will need a Gemini API Key to power the agents. Just add it to a local environment file.
```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

### 3. Run the Ecosystem

You'll need two terminal windows to run the full experience.

**Terminal 1 (Start the Backend Engine):**
```bash
# This starts the unified multi-agent FastAPI service
python -m uvicorn unified_service:app --port 8081 --reload
```

**Terminal 2 (Start your choice of UI):**

*Option A: CLI Orchestrator*
```bash
cd orchestrator
python main.py
```

*Option B: Web UI*
```bash
cd front-end
npm install
npm run dev
```

*(Once running, just follow the prompts and feed it your CSV file—either locally or via a URL!)*

---

## What Makes It Stand Out?

The system doesn't rely on generic hardcoded lists of rules. It adapts to your unique data footprint:

| When the Agent Sees... | It Actively Decides to... |
| :--- | :--- |
| A column with **20% missing values** + **high skewness** | Fill the missing gaps using the **median** |
| A categorical column with **3 unique values** | Apply **One-Hot Encoding** |
| A categorical column with **500 unique values** | Apply **Frequency Encoding** to prevent explosion |
| Extraneous **outliers (15%+)** in a feature | Cap the extremes cleanly using **IQR bounds** |
| Significant **Class imbalance** | Automatically balance classes using **SMOTE** |
| Base Accuracy hovering below your target | Trigger **Optuna hyperparameter tuning** |

---

## What You Get At The End

When the pipeline finishes, AutoML generates a rich output directory (e.g., `autoeda_output_1234567890/`) tailored to your project. Everything is organized and ready to share:

- `data.csv`: A copy of your original dataset for reference.
- `analysis.py`: Your robust, standalone, ready-to-run Python training pipeline.
- `reports/`: Complete classification and insight reports.
- `plots/`: Highly detailed confusion matrices, heatmaps, and visualizations.
- `models/`: Pickles/binaries of the intelligently trained models.

### Demonstrated Success
*(The repo includes out-of-the-box examples that highlight the system's adaptability)*

- **Auto MPG (Regression):** Handled 398 rows x 9 columns to output a Tuned Ensemble model (Score: ~0.90) in ~3m 44s.
- **Breast Cancer (Classification):** Analyzed 699 rows x 11 columns to build a Random Forest model (Score: ~0.97) in ~3m 20s.
- **Diamonds (Regression):** Processed an enormous 53,940 rows x 10 columns to achieve a killer Ensemble model (Score: ~0.98) in just ~19s.

---

## License

This project is licensed under the **MIT License**. Build with it, break it, and make it your own!
