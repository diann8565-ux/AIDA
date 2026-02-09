import pandas as pd
import os

file_path = r"c:\Users\eka\Downloads\aida\test.xlsx"

if os.path.exists(file_path):
    print(f"Inspecting {file_path}...")
    try:
        df = pd.read_excel(file_path)
        print("Columns:", df.columns.tolist())
        print("\nData Types:")
        print(df.dtypes)
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check specific columns for non-numeric types
        for col in ['MonthlyCharges', 'TotalCharges']:
            if col in df.columns:
                print(f"\nChecking {col} values:")
                for i, val in enumerate(df[col].head(10)):
                    print(f"Row {i}: {val} (Type: {type(val)})")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File not found.")
