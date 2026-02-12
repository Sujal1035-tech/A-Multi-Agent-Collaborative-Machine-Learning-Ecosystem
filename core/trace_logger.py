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

        # Step 1: Analysis
        analysis = self.steps.get("analysis", {}).get("analysis_summary", {})
        if analysis:
            lines.append("## Step 1: Analysis Agent\n")
            shape = analysis.get("shape", [])
            lines.append(f"- **Shape:** {shape[0]} rows × {shape[1]} columns" if len(shape) == 2 else "")
            lines.append(f"- **Target:** `{analysis.get('target_column', '?')}`")
            lines.append(f"- **Numerical columns:** {', '.join(analysis.get('data_types', {}).get('numerical', []))}")
            lines.append(f"- **Categorical columns:** {', '.join(analysis.get('data_types', {}).get('categorical', [])) or 'None'}\n")

            # Descriptive statistics table
            desc_stats = analysis.get("descriptive_stats", {})
            if desc_stats:
                lines.append("### Descriptive Statistics\n")
                lines.append("| Column | Mean | Median | Std | Min | Max |")
                lines.append("|--------|------|--------|-----|-----|-----|")
                for col, stats in desc_stats.items():
                    lines.append(f"| {col} | {stats['mean']} | {stats['median']} | {stats['std']} | {stats['min']} | {stats['max']} |")
                lines.append("")

            # Categorical value counts
            cat_stats = analysis.get("categorical_stats", {})
            if cat_stats:
                lines.append("### Categorical Value Counts\n")
                for col, counts in cat_stats.items():
                    lines.append(f"**{col}:** " + ", ".join(f"{k} ({v})" for k, v in counts.items()))
                lines.append("")

            # Nulls table
            missing = analysis.get("missing", {})
            has_nulls = any(v.get("count", 0) > 0 for v in missing.values() if isinstance(v, dict))
            if has_nulls:
                lines.append("### Missing Values\n")
                lines.append("| Column | Count | Percent |")
                lines.append("|--------|-------|---------|")
                for col, info in missing.items():
                    if isinstance(info, dict) and info.get("count", 0) > 0:
                        lines.append(f"| {col} | {info['count']} | {info['percent']}% |")
                lines.append("")

            # Outliers table
            outliers = analysis.get("outliers", {})
            if outliers:
                lines.append("### Outliers (IQR)\n")
                lines.append("| Column | Count | Percent | Bounds |")
                lines.append("|--------|-------|---------|--------|")
                for col, info in outliers.items():
                    if isinstance(info, dict) and info.get("count", 0) > 0:
                        lines.append(f"| {col} | {info['count']} | {info['percent']}% | [{info['lower_bound']}, {info['upper_bound']}] |")
                lines.append("")

            # Skewness
            skewness = analysis.get("skewness", {})
            if skewness:
                lines.append("### Skewness\n")
                lines.append("| Column | Skewness | Interpretation |")
                lines.append("|--------|----------|----------------|")
                for col, val in skewness.items():
                    interp = "Normal" if abs(val) < 0.5 else ("Moderate" if abs(val) < 1 else "High")
                    lines.append(f"| {col} | {val} | {interp} |")
                lines.append("")

        # Step 2: Insights (first pass)
        insights_1 = self.steps.get("insights_1", {})
        if insights_1:
            lines.append("---\n")
            lines.append("## Step 2: Insight Agent (First Pass)\n")
            insight_text = insights_1.get("insights", "")
            # Show first 500 chars to keep it brief
            preview = insight_text[:500] + ("..." if len(insight_text) > 500 else "")
            lines.append(f"{preview}\n")

        # Step 3: Preprocessing
        prep = self.steps.get("preprocessing", {})
        if prep:
            lines.append("---\n")
            lines.append("## Step 3: Preprocessing Agent (LLM Decision)\n")
            strategy = prep.get("preprocessing_strategy", {})

            null_strat = strategy.get("null_strategy", {})
            if null_strat:
                lines.append("### Null Handling Strategy\n")
                lines.append("| Column | Method | Reason |")
                lines.append("|--------|--------|--------|")
                for col, config in null_strat.items():
                    method = config.get("method", config) if isinstance(config, dict) else config
                    reason = config.get("reason", "") if isinstance(config, dict) else ""
                    lines.append(f"| {col} | {method} | {reason} |")
                lines.append("")

            outlier_strat = strategy.get("outlier_strategy", {})
            if outlier_strat:
                method = outlier_strat.get("method", "?")
                threshold = outlier_strat.get("threshold", "?")
                cols = outlier_strat.get("columns", [])
                lines.append(f"### Outlier Strategy\n")
                lines.append(f"- **Method:** {method}")
                lines.append(f"- **Threshold:** {threshold}")
                lines.append(f"- **Columns:** {', '.join(cols) if cols else 'None'}")
                reason = outlier_strat.get("reason", "")
                if reason:
                    lines.append(f"- **Reason:** {reason}")
                lines.append("")

            scaling_strat = strategy.get("scaling_strategy", {})
            if scaling_strat:
                lines.append(f"### Scaling Strategy\n")
                lines.append(f"- **Method:** {scaling_strat.get('method', '?')}")
                lines.append(f"- **Columns:** {', '.join(scaling_strat.get('columns', []))}")
                reason = scaling_strat.get("reason", "")
                if reason:
                    lines.append(f"- **Reason:** {reason}")
                lines.append("")

        # Step 4: Feature Engineering
        feat = self.steps.get("feature", {})
        if feat:
            lines.append("---\n")
            lines.append("## Step 4: Feature Engineering Agent (LLM Decision)\n")
            strategy = feat.get("feature_strategy", {})
            encoding = strategy.get("encoding_strategy", {})

            if encoding:
                lines.append("### Encoding Strategy\n")
                lines.append("| Encoding Type | Columns |")
                lines.append("|---------------|---------|")
                for enc_type in ["onehot", "label", "target"]:
                    cols = encoding.get(enc_type, [])
                    if cols:
                        lines.append(f"| {enc_type} | {', '.join(cols)} |")
                lines.append("")
                enc_reason = encoding.get("reason", "")
                if enc_reason:
                    lines.append(f"**Reasoning:** {enc_reason}\n")

            dropped = strategy.get("features_to_drop", [])
            if dropped:
                lines.append(f"### Features Dropped\n")
                lines.append(f"- {', '.join(dropped)}")
                drop_reason = strategy.get("drop_reason", "")
                if drop_reason:
                    lines.append(f"- **Reason:** {drop_reason}")
                lines.append("")

        # Step 5: Model Training
        models = self.steps.get("models", {})
        if models:
            lines.append("---\n")
            lines.append("## Step 5: Model Training Agent\n")
            lines.append(f"- **Problem Type:** {models.get('problem_type', '?')}")
            lines.append(f"- **Metric:** {models.get('metric', '?')}")
            lines.append(f"- **Used SMOTE:** {'Yes' if models.get('used_balancing') else 'No'}")
            lines.append(f"- **Used Optuna Tuning:** {'Yes' if models.get('used_tuning') else 'No'}\n")

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
            lines.append(f"{evaluation.get('evaluation', 'No evaluation data')}\n")

        # Step 7: Final Insights
        insights_2 = self.steps.get("insights_2", {})
        if insights_2:
            lines.append("---\n")
            lines.append("## Step 7: Final Insights\n")
            insight_text = insights_2.get("insights", "")
            preview = insight_text[:800] + ("..." if len(insight_text) > 800 else "")
            lines.append(f"{preview}\n")

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
