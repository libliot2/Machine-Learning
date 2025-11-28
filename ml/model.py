import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold  # 新增引用
import os

def model_train(
    model_name: str,
    output_path: str = '../model',
    datasets: dict = None,
    perform_cv: bool = False,  # 新增参数：是否执行交叉验证
    cv_folds: int = 5          # 新增参数：折数
):
    print(f"\n--- Training {model_name} ---")
    model = train_algo_wrapper(model_name)
    
    # 获取训练集
    X_train, y_train = datasets['Train']
    
    # 1. 如果需要，先执行交叉验证 (Cross-Validation)
    if perform_cv:
        perform_cross_validation(model, X_train, y_train, k=cv_folds)

    # 2. 在全量训练集上进行最终训练
    print(f"Fitting model on full training set ({len(X_train)} samples)...")
    model.fit(X_train, y_train)
    
    # 3. 保存模型
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    model_file = f"{model_name}_model.pkl"
    save_loc = os.path.join(output_path, model_file)
    joblib.dump(model, save_loc)
    print(f"Model {model_name} saved to {model_file}")

def perform_cross_validation(model, X, y, k=5):
    """
    执行 K-Fold Stratified Cross-Validation 并打印统计结果
    """
    print(f"Running {k}-Fold Cross-Validation...")
    
    # 使用 StratifiedKFold 确保每一折中类别比例一致 (对不平衡数据很重要)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # 计算 Accuracy (n_jobs=-1 使用所有CPU核心加速)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"   > CV Scores: {np.round(scores, 4)}")
    print(f"   > Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print("-" * 30)

def train_algo_wrapper(algo_name: str):
    if algo_name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=42)
    elif algo_name == 'LogisticRegression':
        # 增加 n_jobs=-1 加速训练
        return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1) 
    elif algo_name == 'SVM':
        # SVM 在大数据集上非常慢，建议小心使用
        return SVC(random_state=42)
    elif algo_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif algo_name == 'NeuralNetwork':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    elif algo_name == 'XGBoost':
        print("Initializing XGBoost with Histogram optimization...")
        return XGBClassifier(
            n_estimators=200,       # 树的数量，比 RF 多一点通常更好
            learning_rate=0.1,      # 学习率
            max_depth=10,           # 树深，防止过拟合
            objective='multi:softprob', 
            n_jobs=-1,              # 并行计算
            random_state=42,
            tree_method='hist',     # [关键] 针对大数据集的直方图优化算法，速度极快
            device='cuda'            # 如果你有 GPU，可以改为 'cuda'
        )
    elif algo_name == 'ExtraTrees':
        print("Initializing ExtraTreesClassifier...")
        return ExtraTreesClassifier(
            n_estimators=200,       # 树的数量，200 棵通常足够稳健
            max_depth=None,         # ExtraTrees 通常不需要像 XGBoost 那样严格限制深度
            min_samples_split=5,    # 稍微限制一下叶子节点，防止过分生长
            random_state=42,
            n_jobs=-1               # 并行计算，速度飞快
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")