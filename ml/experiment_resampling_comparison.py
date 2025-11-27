import pandas as pd
import numpy as np
import os
import sys
import time
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

# 确保能导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import train_algo_wrapper
from model_eval import evaluate_metrics

def run_comprehensive_experiment(
    data_path='../data/covtype_processed.csv',
    sample_limit=None  # 💡 限制训练样本数以加速实验 (设为 None 则跑全量)
):
    print("=" * 60)
    print(">>> 综合实验：多模型 vs. 重采样策略 (Class Imbalance)")
    print("=" * 60)

    # 1. 加载数据
    if not os.path.exists(data_path):
        print(f"[错误] 找不到文件: {data_path}")
        return
    
    print(f"Loading Data: {data_path} ...")
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 2. 划分数据集 (保持统一的测试集)
    # stratify=y 确保测试集和训练集的类别分布与原始数据一致
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 💡 如果开启了采样限制，从训练集中再抽取一部分
    if sample_limit and sample_limit < len(X_train_full):
        print(f"\n[注意] 为了加速实验，仅使用 {sample_limit} 条训练样本。")
        # 再次分层采样
        sample_indices = np.random.choice(len(X_train_full), sample_limit, replace=False)
        # 这里的简单随机抽样可能会破坏分布，严谨做法是用 train_test_split 再切一次
        X_train_sub, _, y_train_sub, _ = train_test_split(
            X_train_full, y_train_full, train_size=sample_limit, random_state=42, stratify=y_train_full
        )
        X_train, y_train = X_train_sub, y_train_sub
    else:
        print(f"\n[注意] 使用全量训练集 ({len(X_train_full)} samples)。请耐心等待。")
        X_train, y_train = X_train_full, y_train_full

    # 3. 定义实验配置
    models_to_test = ['LogisticRegression', 'DecisionTree', 'RandomForest']
    
    strategies = {
        'Baseline (Raw)': None,
        'UnderSampling': RandomUnderSampler(random_state=42),
        'OverSampling (SMOTE)': SMOTE(random_state=42)
    }

    results = []

    # 4. 开始循环实验
    for model_name in models_to_test:
        print(f"\n" + "-"*30)
        print(f"🤖 当前模型: {model_name}")
        print("-"*30)

        for strategy_name, sampler in strategies.items():
            print(f"   > 策略: {strategy_name} ...", end=" ")
            start_time = time.time()

            # (A) 重采样 (仅针对当前这一轮的 X_train)
            X_res, y_res = X_train, y_train
            if sampler is not None:
                try:
                    X_res, y_res = sampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"[Skipped] Resampling failed: {e}")
                    continue
            
            # (B) 训练模型
            # 重新初始化模型，确保干净的状态
            clf = train_algo_wrapper(model_name)
            clf.fit(X_res, y_res)

            # (C) 预测 (必须在原始纯净的 X_test 上)
            y_pred = clf.predict(X_test)
            
            # 尝试获取概率用于计算 AUC (如果支持)
            y_proba = None
            if hasattr(clf, "predict_proba"):
                try:
                    y_proba = clf.predict_proba(X_test)
                except:
                    pass
            
            # 如果没有概率，用预测标签代替 (AUC 计算会不准，但为了代码不崩)
            if y_proba is None:
                y_proba = y_pred

            # (D) 计算指标
            # 我们直接调用 sklearn 的函数计算需要的特定指标，比调用 evaluate_metrics 更灵活
            from sklearn.metrics import accuracy_score, f1_score, recall_score
            
            acc = accuracy_score(y_test, y_pred)
            # 关注 Macro Average (宏平均)，这对不平衡类别最重要
            f1_macro = f1_score(y_test, y_pred, average='macro')
            recall_macro = recall_score(y_test, y_pred, average='macro')
            
            elapsed = time.time() - start_time
            print(f"完成 ({elapsed:.1f}s) | Acc: {acc:.4f} | F1-Macro: {f1_macro:.4f}")

            results.append({
                'Model': model_name,
                'Strategy': strategy_name,
                'Accuracy': acc,
                'Macro F1': f1_macro,
                'Macro Recall': recall_macro,
                'Time(s)': elapsed
            })

    # 5. 输出最终对比表
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'Strategy':<20} | {'Accuracy':<8} | {'Macro F1':<8} | {'Macro Rec':<8}")
    print("-" * 80)
    
    # 将结果转换为 DataFrame 方便展示 (如果装了 pandas)
    res_df = pd.DataFrame(results)
    # 按模型和 F1 分数排序
    res_df = res_df.sort_values(by=['Model', 'Macro F1'], ascending=[True, False])
    
    for _, row in res_df.iterrows():
        print(f"{row['Model']:<20} | {row['Strategy']:<20} | {row['Accuracy']:.4f}   | {row['Macro F1']:.4f}   | {row['Macro Recall']:.4f}")
    
    print("="*80)
    
    # 6. 保存结果到 CSV，方便写报告用
    res_df.to_csv('../images/experiment_resampling_results.csv', index=False)
    print("实验结果已保存至 ../images/experiment_resampling_results.csv")

if __name__ == "__main__":
    # 建议先用 50000 样本跑通，确认无误后再设为 None 跑全量
    run_comprehensive_experiment(sample_limit=None)