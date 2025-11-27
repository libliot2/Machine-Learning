import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_advanced_poly_experiment(data_path='../data/covtype_processed.csv'):
    print("=" * 70)
    print(">>> Advanced Open-ended Exploration: Selective Polynomial Features")
    print("    Strategy: Only expand continuous features, keep binary features as is.")
    print("=" * 70)

    # 1. 加载数据
    if not os.path.exists(data_path):
        print(f"[Error] Data file not found: {data_path}")
        return

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. 定义特征组
    # 根据 Covertype 数据集定义，前10列是连续变量
    continuous_cols = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
    ]
    
    # 其余为二值变量 (Soil_Type, Wilderness_Area) 和 目标变量
    # 注意：我们要排除最后一列 Cover_Type
    all_features = df.columns[:-1]
    binary_cols = [col for col in all_features if col not in continuous_cols]
    
    X = df[all_features]
    y = df['Cover_Type']
    
    print(f"Feature Info: {len(continuous_cols)} Continuous, {len(binary_cols)} Binary.")

    # 3. 数据切分 (保持与主程序一致的 random_state 以便对比)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ==========================================================================
    # 实验 A: Baseline (标准逻辑回归)
    # ==========================================================================
    print("\n[Baseline] Training Standard Logistic Regression...")
    start_time = time.time()
    
    # n_jobs=-1 使用所有CPU核心加速
    baseline_model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    baseline_model.fit(X_train, y_train)
    
    base_pred = baseline_model.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    base_time = time.time() - start_time
    
    print(f"   -> Baseline Accuracy: {base_acc:.4f} (Time: {base_time:.2f}s)")

    # ==========================================================================
    # 实验 B: Advanced Poly (仅扩展连续特征)
    # ==========================================================================
    print("\n[Advanced Poly] Training Selective Polynomial Logistic Regression...")
    print("   Pipeline: [Continuous -> Poly(deg=2) -> Scale] + [Binary -> Pass] -> LogReg")

    # 定义转换器
    # 1. 对连续特征：做2次多项式扩展，然后必须重新标准化 (StandardScaler)
    poly_transformer = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # 2. 使用 ColumnTransformer 组合：连续特征走 poly_transformer，二值特征直接透传 (passthrough)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_poly', poly_transformer, continuous_cols),
            ('cat_pass', 'passthrough', binary_cols)
        ]
    )

    # 3. 最终的模型 Pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42))
    ])

    start_time = time.time()
    model_pipeline.fit(X_train, y_train)
    poly_time = time.time() - start_time # 包含特征转换时间

    poly_pred = model_pipeline.predict(X_test)
    poly_acc = accuracy_score(y_test, poly_pred)

    # ==========================================================================
    # 结果分析与维度计算
    # ==========================================================================
    print(f"   -> Poly Model Accuracy: {poly_acc:.4f} (Time: {poly_time:.2f}s)")

    # 计算特征数量的变化
    n_cont = len(continuous_cols)
    # n_poly = n_cont + (n_cont * (n_cont+1)) / 2  (C(n+d, d) - 1)
    n_poly_features = int(n_cont + (n_cont * (n_cont + 1)) / 2) # 10 -> 65
    total_features_new = n_poly_features + len(binary_cols)

    print("\n" + "=" * 70)
    print(">>> Comparative Analysis")
    print("=" * 70)
    print(f"Original Feature Count : {X.shape[1]}")
    print(f"Expanded Feature Count : {total_features_new} (Approx. {total_features_new - X.shape[1]} interaction features added)")
    print("-" * 30)
    print(f"Baseline Accuracy      : {base_acc:.4f}")
    print(f"Poly Pipeline Accuracy : {poly_acc:.4f}")
    
    improvement = poly_acc - base_acc
    print(f"Improvement            : {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0.001:
        print("\n[CONCLUSION] Strategy Successful.")
        print("Explicitly modeling non-linear interactions between continuous variables (e.g., Elevation * Hydrology)")
        print("improved the linear model's performance without incurring the cost of full polynomial expansion.")
    else:
        print("\n[CONCLUSION] Performance gain is minimal.")
        print("This strongly suggests that the underlying patterns are 'threshold-based' rather than smooth curves.")
        print("Recommendation: Move to Tree-based models (Random Forest / XGBoost) for significant gains.")

if __name__ == "__main__":
    run_advanced_poly_experiment()