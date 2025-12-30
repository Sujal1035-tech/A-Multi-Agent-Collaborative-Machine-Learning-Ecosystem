import os, shutil

def write_project(folder, csv_path, analysis_code, readme, insights=None):
    os.makedirs(folder, exist_ok=True)
    for d in ["stats", "plots", "reports"]:
        os.makedirs(f"{folder}/{d}", exist_ok=True)

    shutil.copy(csv_path, f"{folder}/data.csv")

    open(f"{folder}/analysis.py", "w").write(analysis_code)
    open(f"{folder}/README.md", "w").write(readme)
    
    # Save insights if provided
    if insights:
        open(f"{folder}/reports/insights.txt", "w").write(insights)
