import pandas as pd
import numpy as np

def verify_processed_data(filepath='covtype_processed.csv'):
    print(f"Running Verification on {filepath}...\n")
    print("-" * 50)
    
    # 1. 加载数据
    try:
        df = pd.read_csv(filepath)
        print(df.head())
        print(f"[PASS] File loaded successfully.")
    except FileNotFoundError:
        print(f"[FAIL] File not found: {filepath}")
        return

    expected_shape = (581012, 55)
    if df.shape == expected_shape:
        print(f"[PASS] Data Shape is correct: {df.shape}")
    else:
        print(f"[FAIL] Incorrect Shape: {df.shape}, expected {expected_shape}")

    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print(f"[PASS] No missing values found.")
    else:
        print(f"[FAIL] Found {missing_count} missing values!")

    numerical_cols = df.columns[:10]
    means = df[numerical_cols].mean()
    stds = df[numerical_cols].std()
    
    is_standardized = (means.abs() < 0.01).all() and (stds.between(0.99, 1.01)).all()
    
    if is_standardized:
        print(f"[PASS] First 10 numerical features are properly standardized (Mean ~0, Std ~1).")
    else:
        print(f"[WARNING] Numerical features might not be standardized correctly.")
        print(f"      Mean range: [{means.min():.4f}, {means.max():.4f}]")
        print(f"      Std range:  [{stds.min():.4f}, {stds.max():.4f}]")

    binary_sample_col = df.columns[10] 
    unique_vals = df[binary_sample_col].unique()
    
    valid_binary = np.all([x in [0, 1] for x in unique_vals])
    
    if valid_binary:
        print(f"[PASS] Binary features (e.g., {binary_sample_col}) preserved correctly (values: {unique_vals}).")
    else:
        print(f"[FAIL] Binary features corrupted! Found values: {unique_vals}")
        print("      Did you accidentally standardize the One-Hot encoded columns?")

    target_col = df.columns[-1]
    expected_classes = {1, 2, 3, 4, 5, 6, 7}
    actual_classes = set(df[target_col].unique())
    
    if actual_classes == expected_classes:
        print(f"[PASS] Target variable '{target_col}' contains correct classes: {actual_classes}.")
    else:
        print(f"[FAIL] Unexpected classes in target: {actual_classes}")

    print("-" * 50)
    print("Verification Complete.")

if __name__ == "__main__":
    verify_processed_data()