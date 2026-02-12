def ask_permission(folder: str) -> bool:
    print(f"\nCreate project in '{folder}'?")
    print("Files: analysis.py, README.md, insights.txt, pipeline_trace.md, plots/")
    return input("Proceed? (yes/no): ").lower() == "yes"
