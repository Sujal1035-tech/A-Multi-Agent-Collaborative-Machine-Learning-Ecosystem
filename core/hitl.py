def ask_permission(folder: str) -> bool:
    print(f"\nCreate project in '{folder}'?")
    print("Files: analysis.py, README.md, insights.txt, pipeline_trace.md, plots/")
    
    # Clear the input buffer before asking, in case the user pressed Enter while waiting
    import sys
    if sys.platform == 'win32':
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    else:
        import select
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()

    return input("Proceed? (yes/no): ").strip().lower() == "yes"
