import os
import pandas as pd
import numpy as np

def clean_heat_exchanger_data(input_path: str, output_path: str):
    print("=" * 60)
    print("Cleaning Heat Exchanger Fouling Dataset")
    print("=" * 60)
    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Step 1: Drop duplicate .1 columns
    dup_cols = [c for c in df.columns if c.endswith('.1')]
    df.drop(columns=dup_cols, inplace=True)
    print(f"Dropped {len(dup_cols)} duplicate (.1) columns")

    # Step 2: Drop zero-variance columns
    zero_var_cols = [c for c in df.columns if df[c].var() < 1e-10]
    df.drop(columns=zero_var_cols, inplace=True)
    print(f"Dropped {len(zero_var_cols)} zero-variance constant columns")

    # Step 3: Verify no nulls
    null_count = df.isnull().sum().sum()
    print(f"Missing values: {null_count}")

    print(f"Final cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    candidate_inputs = [
        r"c:\Users\riddh\OneDrive\Desktop\industrial_agent\heat_exchanger_fouling_dataset.csv",
        r"C:\Users\riddh\Downloads\heat_exchanger_fouling_dataset.csv",
        r"C:\Projects\industrial_agent\heat_exchanger_fouling_dataset.csv",
    ]
    raw_path = None
    for p in candidate_inputs:
        if os.path.exists(p):
            raw_path = p
            break

    out_path = r"c:\Users\riddh\OneDrive\Desktop\industrial_agent\heat_exchanger_fouling_clean.csv"

    if raw_path:
        print(f"Using input dataset: {raw_path}")
        clean_heat_exchanger_data(raw_path, out_path)
    else:
        print("Input file not found in candidates:", candidate_inputs)

