def ask_permission(folder: str) -> bool:
    print(f"\nCreate project in '{folder}'?")
    print("Files: analysis.py, README.md, stats/, plots/, reports/")
    return input("Proceed? (yes/no): ").lower() == "yes"
