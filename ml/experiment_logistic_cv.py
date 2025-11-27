import pandas as pd
import os
import sys

# 确保能导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import train_algo_wrapper, perform_cross_validation

def run_logistic_validation(data_path='../data/covtype_processed.csv'):
    print("=" * 50)
    print(">>> 独立实验：逻辑回归泛化性验证 (Cross-Validation)")
    print("=" * 50)

    # 1. 加载数据
    if not os.path.exists(data_path):
        print(f"[错误] 找不到数据文件: {data_path}")
        print("请先运行 main.py 或 data_preprocessing.py 生成处理后的数据。")
        return

    print(f"正在加载数据: {data_path} ...")
    # 建议使用 float32 节省内存，虽然逻辑回归对内存要求不高
    df = pd.read_csv(data_path)
    print(f"数据加载完成，形状: {df.shape}")

    # 2. 准备 X 和 y
    # 假设最后一列是标签 'Cover_Type'
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 3. 初始化模型
    print("\n正在初始化 LogisticRegression (n_jobs=-1)...")
    model = train_algo_wrapper('LogisticRegression')

    # 4. 执行交叉验证
    # 这里直接调用我们在 model.py 里封装好的函数
    print("\n开始执行 5-Fold Cross-Validation...")
    perform_cross_validation(model, X, y, k=5)

    print("\n" + "=" * 50)
    print("实验结束。如果 Mean Accuracy 与 Test Set 结果接近且方差极小，")
    print("则可强有力地证明模型具有良好的泛化能力（未过拟合）。")
    print("=" * 50)

if __name__ == "__main__":
    # 确保路径相对于 ml/ 目录是正确的
    run_logistic_validation()