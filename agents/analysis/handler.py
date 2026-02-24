"""
Analysis Agent Handler
Shared logic for analysis functionality
"""

import pandas as pd
import os
from a2a.schemas import A2ATask, A2AResponse


def generate_eda_plots(df, target_col, output_folder, log_callback=None):
    """Generate EDA plots and save to plots/ subfolder."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    saved = []

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def log(msg):
        if log_callback:
            log_callback(msg)

    # 1. Correlation Heatmap
    if len(numerical_cols) >= 2:
        try:
            plt.figure(figsize=(10, 8))
            corr = df[numerical_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, square=True, linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            path = os.path.join(plots_dir, 'correlation_heatmap.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved.append('correlation_heatmap.png')
            log('[ANALYSIS] Saved correlation_heatmap.png')
        except Exception as e:
            log(f'[ANALYSIS] Skipped heatmap: {e}')

    # 2. Target Distribution
    if target_col in df.columns:
        try:
            plt.figure(figsize=(8, 5))
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
                df[target_col].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
                plt.ylabel('Count')
            else:
                df[target_col].hist(bins=30, color='steelblue', edgecolor='black')
                plt.ylabel('Frequency')
            plt.title(f'Target Distribution: {target_col}')
            plt.xlabel(target_col)
            plt.tight_layout()
            path = os.path.join(plots_dir, 'target_distribution.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved.append('target_distribution.png')
            log('[ANALYSIS] Saved target_distribution.png')
        except Exception as e:
            log(f'[ANALYSIS] Skipped target plot: {e}')

    # 3. Feature Distributions (grid)
    plot_cols = [c for c in numerical_cols if c != target_col][:12]  # Max 12
    if plot_cols:
        try:
            ncols = min(3, len(plot_cols))
            nrows = (len(plot_cols) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            if nrows * ncols == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, col in enumerate(plot_cols):
                df[col].hist(bins=25, ax=axes[i], color='teal', edgecolor='black', alpha=0.7)
                axes[i].set_title(col)
                axes[i].set_xlabel('')
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle('Feature Distributions', fontsize=14, y=1.02)
            plt.tight_layout()
            path = os.path.join(plots_dir, 'feature_distributions.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved.append('feature_distributions.png')
            log('[ANALYSIS] Saved feature_distributions.png')
        except Exception as e:
            log(f'[ANALYSIS] Skipped distributions: {e}')

    # 4. Box Plots (outlier visualization)
    if plot_cols:
        try:
            ncols = min(3, len(plot_cols))
            nrows = (len(plot_cols) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            if nrows * ncols == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, col in enumerate(plot_cols):
                df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle('Box Plots (Outlier Detection)', fontsize=14, y=1.02)
            plt.tight_layout()
            path = os.path.join(plots_dir, 'box_plots.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved.append('box_plots.png')
            log('[ANALYSIS] Saved box_plots.png')
        except Exception as e:
            log(f'[ANALYSIS] Skipped box plots: {e}')

    return saved


def analyze(df: pd.DataFrame, target_column: str = None):
    """Enhanced analysis with detailed stats for smart preprocessing"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    cardinality = {col: df[col].nunique() for col in categorical_cols}
    
    # Use provided target column or fall back to last column
    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        target_col = df.columns[-1]
    
    # Enhanced null analysis
    null_info = {}
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        null_info[col] = {"count": int(null_count), "percent": round(null_pct, 2)}
    
    # Outlier detection using IQR for numerical columns
    outlier_info = {}
    for col in numerical_cols:
        if col == target_col:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_pct = (outlier_count / len(df)) * 100
        outlier_info[col] = {
            "count": int(outlier_count), 
            "percent": round(outlier_pct, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2)
        }
    
    # Skewness for numerical columns
    skewness = {}
    for col in numerical_cols:
        try:
            skew_val = df[col].skew()
            skewness[col] = round(skew_val, 2)
        except Exception:
            skewness[col] = 0
    
    # Cardinality for all columns (for encoding decisions)
    all_cardinality = {col: int(df[col].nunique()) for col in df.columns}
    
    # Descriptive statistics for numerical columns
    descriptive_stats = {}
    for col in numerical_cols:
        descriptive_stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "median": round(float(df[col].median()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "q1": round(float(df[col].quantile(0.25)), 4),
            "q3": round(float(df[col].quantile(0.75)), 4),
        }
    
    # Value counts for categorical columns (top 10)
    categorical_stats = {}
    for col in categorical_cols:
        counts = df[col].value_counts().head(10)
        categorical_stats[col] = {str(k): int(v) for k, v in counts.items()}
    
    return {
        "columns": df.columns.tolist(),
        "shape": list(df.shape),
        "missing": null_info,
        "data_types": {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        },
        "cardinality": all_cardinality,
        "target_column": target_col,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "outliers": outlier_info,
        "skewness": skewness,
        "descriptive_stats": descriptive_stats,
        "categorical_stats": categorical_stats
    }


def handle_analysis(task: A2ATask, log_callback=None) -> A2AResponse:
    """Handle analysis task"""
    if log_callback:
        log_callback(f"[ANALYSIS] Starting analysis on {task.input.get('csv_path')}")

    df = pd.read_csv(task.input["csv_path"])
    target_column = task.input.get("target_column")
    result = analyze(df, target_column)

    # Generate EDA plots if output_folder is provided
    output_folder = task.input.get("output_folder")
    plots_saved = []
    if output_folder:
        target_col = result.get("target_column", df.columns[-1])
        # Pass callback to generate_eda_plots if needed, or just let it print (which won't be captured)
        # Better to modify generate_eda_plots too, but for now we'll log high-level
        if log_callback:
            log_callback(f"[ANALYSIS] Generating plots in {output_folder}...")
        
        # We need to update generate_eda_plots to accept callback if we want granular plot logs
        # For now, let's wrap the call
        plots_saved = generate_eda_plots(df, target_col, output_folder, log_callback)

    return A2AResponse(
        task_id=task.task_id,
        sender="analysis-agent",
        status="COMPLETED",
        output={
            "analysis_summary": result,
            "plots": plots_saved
        }
    )
