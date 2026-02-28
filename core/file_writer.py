import os, shutil
import sys
import json
import pandas as pd
from core.data_utils import load_csv_robust


def ask_permission(folder: str) -> bool:
    """Ask user for permission before creating the project folder."""
    print(f"\nCreate project in '{folder}'?")
    print("Files: analysis.py, README.md, insights.txt, pipeline_trace.md, plots/")
    
    # Clear the input buffer before asking, in case the user pressed Enter while waiting
    if sys.platform == 'win32':
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    else:
        import select
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()

    return input("Proceed? (yes/no): ").strip().lower() == "yes"


def write_project(folder, csv_path, analysis_code, readme, insights=None, trace_report=None):
    """Write all project output files to the given folder."""
    os.makedirs(folder, exist_ok=True)

    # Handle both local paths and URLs
    if csv_path.lower().startswith(('http://', 'https://')):
        # Download from URL using pandas
        df = load_csv_robust(csv_path)
        df.to_csv(f"{folder}/data.csv", index=False)
    else:
        # Copy local file
        shutil.copy(csv_path, f"{folder}/data.csv")

    with open(f"{folder}/analysis.py", "w", encoding="utf-8") as f:
        f.write(analysis_code)
    with open(f"{folder}/README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    
    # Save insights if provided
    if insights:
        with open(f"{folder}/insights.txt", "w", encoding="utf-8") as f:
            f.write(insights)

    # Save pipeline trace report
    if trace_report:
        with open(f"{folder}/pipeline_trace.md", "w", encoding="utf-8") as f:
            f.write(trace_report)
