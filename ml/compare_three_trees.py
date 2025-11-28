import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

from data_preprocessing import data_preprocessing
from model import model_train
from model_eval import eval_models

def main(
    raw_dataset_path: str = '../data/covtype.data.gz',
    dataset_path: str = '../data/covtype_processed.csv',
    is_preprocess: bool = False,
):
    print(">>> 启动模型对比实验 (RF vs XGBoost vs ExtraTrees)...")

    # 1. 加载数据
    if is_preprocess:
        data = data_preprocessing(file_name=raw_dataset_path)
    else:
        if os.path.exists(dataset_path):
            print(f"Loading processed data from {dataset_path}...")
            data = pd.read_csv(dataset_path)
        else:
            print("[Error] 找不到数据，请先运行预处理。")
            return

    # 2. 准备数据
    X = data.iloc[:, :-1] 
    y = data.iloc[:, -1]

    # [关键] 统一将标签调整为 0-6 (适应 XGBoost，且兼容 Sklearn)
    print("  -> Adjusting labels (1-7) to (0-6) for consistency.")
    y = y - 1 

    # 3. 划分数据集 (使用 Stratified 保持类别分布)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    datasets = {
        'Train': (X_train, y_train),
        'Test': (X_test, y_test)
    }

    # 4. 定义要对比的模型列表
    models_to_compare = ['RandomForest', 'XGBoost', 'ExtraTrees']

    # 5. 循环训练
    for model_name in models_to_compare:
        model_train(
            model_name=model_name,
            datasets=datasets,
            perform_cv=False # 设为 False 以加快速度，如果写报告需要更严谨可设为 True
        )
    
    # 6. 统一评估并画图
    # model_eval 会读取所有传入的模型，并在同一张图上画出它们的 ROC 曲线
    model_files = [f"../model/{name}_model.pkl" for name in models_to_compare]
    
    print("\n>>> 开始评估与绘图...")
    eval_models(
        datasets=datasets,
        models_path=model_files,
    )

if __name__ == "__main__":
    main()