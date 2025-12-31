"""
User Input Module
Handles interactive input for CSV path and target column selection.
Supports both local file paths and URLs.
"""

import os
import pandas as pd


def is_url(path: str) -> bool:
    """Check if the input is a URL"""
    return path.lower().startswith(('http://', 'https://'))


def load_csv(path_or_url: str) -> pd.DataFrame:
    """
    Load CSV from local path or URL.
    Pandas read_csv supports both automatically.
    """
    try:
        df = pd.read_csv(path_or_url)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")


def get_csv_path() -> tuple[str, pd.DataFrame]:
    """
    Prompt user for CSV path (local or URL) and validate it.
    Returns tuple of (path, dataframe).
    """
    while True:
        print("\n📂 Enter path to CSV file (local path or URL):")
        print("   Examples: D:\\data\\file.csv or https://example.com/data.csv")
        path = input("> ").strip().strip('"').strip("'")
        
        if not path:
            print("❌ Path cannot be empty. Please try again.")
            continue
        
        # Check if it's a URL or local path
        if is_url(path):
            print(f"🌐 Detected URL. Downloading...")
            try:
                df = load_csv(path)
                print(f"✅ Successfully loaded {len(df)} rows from URL")
                return path, df
            except Exception as e:
                print(f"❌ Failed to load URL: {e}")
                print("   Please check the URL and try again.")
                continue
        else:
            # Local path
            if not os.path.exists(path):
                print(f"❌ File not found: {path}")
                print("   Please check the path and try again.")
                continue
            
            if not path.lower().endswith('.csv'):
                print("⚠️  Warning: File doesn't have .csv extension.")
                confirm = input("   Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            try:
                df = load_csv(path)
                print(f"✅ Successfully loaded {len(df)} rows")
                return path, df
            except Exception as e:
                print(f"❌ Failed to read file: {e}")
                continue


def get_target_column(columns: list) -> str:
    """
    Show available columns and prompt user to select target column.
    """
    print("\n📋 Available columns:")
    for i, col in enumerate(columns, 1):
        print(f"   {i}. {col}")
    
    while True:
        print("\n🎯 Enter target column name (or number):")
        choice = input("> ").strip()
        
        if not choice:
            print("❌ Please enter a column name or number.")
            continue
        
        # Check if it's a number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(columns):
                selected = columns[idx]
                print(f"✅ Selected target column: {selected}")
                return selected
            else:
                print(f"❌ Invalid number. Please enter 1-{len(columns)}.")
                continue
        
        # Check if it's a column name
        if choice in columns:
            print(f"✅ Selected target column: {choice}")
            return choice
        
        # Case-insensitive match
        for col in columns:
            if col.lower() == choice.lower():
                print(f"✅ Selected target column: {col}")
                return col
        
        print(f"❌ Column '{choice}' not found. Please try again.")


def get_user_input() -> tuple[str, str, pd.DataFrame]:
    """
    Main function to get user input for CSV path and target column.
    
    Returns:
        tuple: (csv_path, target_column, dataframe)
    """
    print("\n" + "=" * 60)
    print("  📊 AutoEDA - Dataset Configuration")
    print("=" * 60)
    
    # Get CSV path and load data
    csv_path, df = get_csv_path()
    
    # Get target column
    target_column = get_target_column(df.columns.tolist())
    
    print("\n" + "-" * 60)
    print(f"📁 Dataset: {csv_path}")
    print(f"🎯 Target:  {target_column}")
    print(f"📊 Shape:   {df.shape[0]} rows × {df.shape[1]} columns")
    print("-" * 60 + "\n")
    
    return csv_path, target_column, df


if __name__ == "__main__":
    # Test the module
    path, target, df = get_user_input()
    print(f"\nResult: path={path}, target={target}, shape={df.shape}")
