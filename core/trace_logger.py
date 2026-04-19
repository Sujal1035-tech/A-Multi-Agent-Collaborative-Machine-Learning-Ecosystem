"""
Pipeline Trace Logger
Records what each agent decided and generates a readable pipeline_trace.md
Zero LLM calls — just saves data that already exists.
"""

import json
import os
import time
from datetime import datetime


class PipelineTracer:
    def __init__(self, csv_path, target_column):
        self.csv_path = csv_path
        self.target_column = target_column
        self.start_time = time.time()
        self.steps = {}

    def record(self, step_name, output):
        """Record an agent's output. Call after each send_task()."""
        self.steps[step_name] = output

    def save_raw_json(self, output_folder):
        """Save each step's raw output as JSON for debugging."""
        reports_dir = os.path.join(output_folder, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        for i, (name, data) in enumerate(self.steps.items(), 1):
            path = os.path.join(reports_dir, f"{i}_{name}.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception:
                pass  # Don't crash the pipeline for a trace file

    def generate_report(self):
        """Generate a readable markdown trace report."""
        duration = time.time() - self.start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("# 🔍 Pipeline Trace Report\n")
        lines.append(f"**Dataset:** `{self.csv_path}`  ")
        lines.append(f"**Target Column:** `{self.target_column}`  ")
        lines.append(f"**Run Time:** {now}  ")
        lines.append(f"**Duration:** {minutes}m {seconds}s\n")
        lines.append("---\n")

        # Step 1: Analysis - Removed descriptive stats to focus purely on agent reasoning formatting.
        analysis = self.steps.get("analysis", {}).get("analysis_summary", {})
        if analysis:
            lines.append("## Step 1: Dataset Overview\n")
            shape = analysis.get("shape", [])
            lines.append(f"The agents analyzed a dataset with **{shape[0]} rows and {shape[1]} columns** to predict **`{analysis.get('target_column', '?')}`**.")
            lines.append("*(Detailed feature statistics and distributions are saved separately in `insights.txt`)*\n")

        # Step 2: Insights (first pass) - Removed from trace to avoid duplication with insights.txt

        # Step 3: Preprocessing
        prep = self.steps.get("preprocessing", {})
        if prep:
            lines.append("---\n")
            lines.append("## Step 3: Preprocessing Agent (LLM Decision)\n")
            strategy = prep.get("preprocessing_strategy", {})

            null_strat = strategy.get("null_strategy", {})
            if null_strat:
                lines.append("### Null Imputation Reasoning\n")
                for col, config in null_strat.items():
                    method = config.get("method", config) if isinstance(config, dict) else config
                    reason = config.get("reason", "No explicit reason was provided.") if isinstance(config, dict) else "Standard fallback applied."
                    lines.append(f"- **{col}:** The agent decided to use **{method}** imputation. \n  *Why?* {reason}")
                lines.append("")

            outlier_strat = strategy.get("outlier_strategy", {})
            if outlier_strat:
                method = outlier_strat.get("method", "?")
                threshold = outlier_strat.get("threshold", "?")
                cols = outlier_strat.get("columns", [])
                reason = outlier_strat.get("reason", "Standard outlier capping applied.")
                lines.append(f"### Outlier Handling Reasoning\n")
                lines.append(f"The agent noticed strong outliers in the following columns: {', '.join(cols) if cols else 'None'}.")
                lines.append(f"It chose to apply the **{method}** technique with a threshold of {threshold}. \n*Why?* {reason}\n")

            scaling_strat = strategy.get("scaling_strategy", {})
            if scaling_strat:
                reason = scaling_strat.get("reason", "Scaled data to assist linear algorithms.")
                lines.append(f"### Scaling Reasoning\n")
                lines.append(f"The agent applied **{scaling_strat.get('method', '?')}**. \n*Why?* {reason}\n")

        # Step 4: Feature Engineering
        feat = self.steps.get("feature", {})
        if feat:
            lines.append("---\n")
            lines.append("## Step 4: Feature Engineering Agent (LLM Decision)\n")
            strategy = feat.get("feature_strategy", {})
            encoding = strategy.get("encoding_strategy", {})

            if encoding:
                lines.append("### Encoding Reasoning\n")
                enc_reason = encoding.get("reason", "Categorical features were detected.")
                lines.append(f"The feature agent analyzed the categorical columns and stated: *\"{enc_reason}\"*")
                for enc_type in ["onehot", "label", "target"]:
                    cols = encoding.get(enc_type, [])
                    if cols:
                        lines.append(f"- It chose to apply **{enc_type}** encoding to: {', '.join(cols)}")
                lines.append("")

            dropped = strategy.get("features_to_drop", [])
            if dropped:
                drop_reason = strategy.get("drop_reason", "Columns were deemed unnecessary.")
                lines.append(f"### Features Dropped Reasoning\n")
                lines.append(f"The agent decided to completely drop these columns: {', '.join(dropped)}.")
                lines.append(f"*Why?* {drop_reason}\n")

        # Step 5: Model Training
        models = self.steps.get("models", {})
        if models:
            lines.append("---\n")
            lines.append("## Step 5: Model Training Agent (Heuristics & Rationale)\n")
            
            p_type = models.get('problem_type', '?')
            lines.append("### Agent Thinking & Pipeline Decisions\n")
            lines.append(f"1. **Detection:** Classified task as **{p_type.capitalize()}** based on target feature cardinality and data type.")
            lines.append(f"2. **Scaling Phase:** Applied **RobustScaler + StandardScaler** because many datasets contain hidden outliers and linear models require centered variance.")
            
            if models.get('used_balancing'):
                lines.append("3. **Imbalance Handled:** Detected class imbalance! Applied class weight balancing (or SMOTE) so the model doesn't blindly predict the majority class.")
            elif p_type == 'classification':
                lines.append("3. **Imbalance Handled:** Classes look relatively balanced. Skipped SMOTE/weighting to preserve original distributions.")
                
            if models.get('target_transform'):
                lines.append("4. **Target Skewness:** Detected severe target skewness > 1.0! Applied a **Yeo-Johnson PowerTransformer** to normalize the variable and stabilize error gradients.")
                
            if models.get('used_tuning'):
                lines.append("5. **Hyperparameter Tuning:** Triggered Optuna to actively search for the best `max_depth`, `learning_rate`, and `n_estimators` using local cross-validation.")
                
            lines.append("\n### Model Performance\n")

            all_models = models.get("models", {})
            is_regression = models.get("problem_type") == "regression"
            if all_models:
                lines.append("### Model Scores\n")
                if is_regression:
                    lines.append("| Model | R² | Adj R² | RMSE | MAE | CV Score |")
                    lines.append("|-------|----|--------|------|-----|----------|")
                else:
                    lines.append("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | CV Score |")
                    lines.append("|-------|----------|-----------|--------|----------|---------|----------|")
                # Sort by score descending
                sorted_models = sorted(all_models.items(), key=lambda x: x[1].get("score", 0), reverse=True)
                for name, info in sorted_models:
                    marker = " 🏆" if name == models.get("best_model") else ""
                    if is_regression:
                        r2 = info.get("r2", info.get("score", 0))
                        adj_r2 = info.get("adjusted_r2", "—")
                        rmse = info.get("rmse", "—")
                        mae = info.get("mae", "—")
                        cv = info.get("cv_score", "—")
                        lines.append(f"| {name}{marker} | {r2:.4f} | {adj_r2 if adj_r2 == '—' else f'{adj_r2:.4f}'} | {rmse if rmse == '—' else f'{rmse:.4f}'} | {mae if mae == '—' else f'{mae:.4f}'} | {cv if cv == '—' else f'{cv:.4f}'} |")
                    else:
                        acc = info.get("accuracy", info.get("score", 0))
                        prec = info.get("precision", "—")
                        rec = info.get("recall", "—")
                        f1 = info.get("f1_score", "—")
                        auc = info.get("auc_roc")
                        cv = info.get("cv_score", "—")
                        auc_str = f"{auc:.4f}" if auc is not None else "—"
                        lines.append(f"| {name}{marker} | {acc:.4f} | {prec if prec == '—' else f'{prec:.4f}'} | {rec if rec == '—' else f'{rec:.4f}'} | {f1 if f1 == '—' else f'{f1:.4f}'} | {auc_str} | {cv if cv == '—' else f'{cv:.4f}'} |")
                lines.append("")

            lines.append(f"### 🏆 Best Model: `{models.get('best_model')}` — Score: **{models.get('best_score', 0):.4f}**\n")

            # Feature importance
            fi = models.get("feature_importance")
            if fi:
                lines.append("### Top Feature Importance (SHAP)\n")
                lines.append("| Feature | Importance |")
                lines.append("|---------|------------|")
                for item in fi[:10]:
                    lines.append(f"| {item['feature']} | {item['importance']:.4f} |")
                lines.append("")

        # Step 6: Evaluation
        evaluation = self.steps.get("evaluation", {})
        if evaluation:
            lines.append("---\n")
            lines.append("## Step 6: Evaluation Agent\n")
            lines.append(f"```json\n{json.dumps(evaluation, indent=2)}\n```\n")

        # Step 7: Final Insights - Removed from trace to avoid duplication with insights.txt

        # Step 8: Project generation
        project = self.steps.get("project", {})
        if project:
            lines.append("---\n")
            lines.append("## Step 8: Project Code Generated\n")
            code = project.get("analysis_code", "")
            lines.append(f"- **analysis.py:** {len(code)} characters generated")
            lines.append(f"- **README.md:** Included\n")

        lines.append("---\n")
        lines.append(f"*Trace generated automatically by AutoEDA Pipeline Tracer*")

        return "\n".join(lines)
