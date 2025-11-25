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

    # 2. 检查形状 (Shape)
    # Covertype 应为 (581012, 55)
    expected_shape = (581012, 55)
    if df.shape == expected_shape:
        print(f"[PASS] Data Shape is correct: {df.shape}")
    else:
        print(f"[FAIL] Incorrect Shape: {df.shape}, expected {expected_shape}")

    # 3. 检查缺失值 (Missing Values) - Task 1.1 [cite: 34]
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print(f"[PASS] No missing values found.")
    else:
        print(f"[FAIL] Found {missing_count} missing values!")

    # 4. 验证标准化 (Standardization) - Task 1.3 
    # 前10列应该是被缩放过的 (Mean ~0, Std ~1)
    # 我们取前 10 列的名称（假设列顺序未变）
    numerical_cols = df.columns[:10]
    means = df[numerical_cols].mean()
    stds = df[numerical_cols].std()
    
    # 判断标准：均值绝对值小于 0.01，标准差在 0.99-1.01 之间
    is_standardized = (means.abs() < 0.01).all() and (stds.between(0.99, 1.01)).all()
    
    if is_standardized:
        print(f"[PASS] First 10 numerical features are properly standardized (Mean ~0, Std ~1).")
    else:
        print(f"[WARNING] Numerical features might not be standardized correctly.")
        print(f"      Mean range: [{means.min():.4f}, {means.max():.4f}]")
        print(f"      Std range:  [{stds.min():.4f}, {stds.max():.4f}]")

    # 5. 验证二值特征 (Binary Features)
    # 第 11 列到倒数第 2 列 (Wilderness + Soil) 应该只包含 0 或 1
    # 随机抽查一列，比如 'Wilderness_Area_1' (第11列，索引10)
    binary_sample_col = df.columns[10] 
    unique_vals = df[binary_sample_col].unique()
    
    # 允许浮点数的 0.0 和 1.0
    valid_binary = np.all([x in [0, 1] for x in unique_vals])
    
    if valid_binary:
        print(f"[PASS] Binary features (e.g., {binary_sample_col}) preserved correctly (values: {unique_vals}).")
    else:
        print(f"[FAIL] Binary features corrupted! Found values: {unique_vals}")
        print("      Did you accidentally standardize the One-Hot encoded columns?")

    # 6. 验证目标变量 (Target Variable)
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