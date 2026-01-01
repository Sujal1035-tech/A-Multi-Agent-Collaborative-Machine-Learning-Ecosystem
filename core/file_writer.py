import os, shutil
import pandas as pd

def write_project(folder, csv_path, analysis_code, readme, insights=None):
    os.makedirs(folder, exist_ok=True)
    for d in ["stats", "plots", "reports"]:
        os.makedirs(f"{folder}/{d}", exist_ok=True)

    # Handle both local paths and URLs
    if csv_path.lower().startswith(('http://', 'https://')):
        # Download from URL using pandas
        df = pd.read_csv(csv_path)
        df.to_csv(f"{folder}/data.csv", index=False)
    else:
        # Copy local file
        shutil.copy(csv_path, f"{folder}/data.csv")

    open(f"{folder}/analysis.py", "w", encoding="utf-8").write(analysis_code)
    open(f"{folder}/README.md", "w", encoding="utf-8").write(readme)
    
    # Save insights if provided
    if insights:
        open(f"{folder}/reports/insights.txt", "w", encoding="utf-8").write(insights)

