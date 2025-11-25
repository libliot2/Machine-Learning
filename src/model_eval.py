import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ---------------------------
# 1. 读取数据和模型
# ---------------------------
# 1. 读取 CSV 数据集
file_path = 'covtype_processed.csv'  # 请替换为你的 CSV 文件路径
data = pd.read_csv(file_path)

def convert_to_float_if_needed(df):
    """
    检查 DataFrame 的每一列，如果数据类型为 object（通常是字符串）或整数，
    则尝试将其转换为浮点数。
    """
    for col in df.columns:
        # 检查是否为非浮点数类型
        if not pd.api.types.is_float_dtype(df[col]):
            original_dtype = df[col].dtype
            try:
                # 尝试转换为浮点数
                df[col] = pd.to_numeric(df[col], errors='raise').astype(np.float32)
                if original_dtype != df[col].dtype:
                    print(f"  - 列 {col} 已从 {original_dtype} 转换为 float32")
            except (ValueError, TypeError) as e:
                print(f"  - 警告: 列 {col} 无法转换为数字 (原类型: {original_dtype})，错误: {e}")
                # 如果转换失败，可以选择跳过该列或中断程序
                # 此处选择中断，因为无法处理非数字特征
                raise ValueError(f"列 {col} 包含无法转换的非数字数据，无法继续。")
    return df

data = convert_to_float_if_needed(data[1:])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分数据集（必须和训练时一致，包括 random_state 和 stratify）
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 加载模型
dt_model = joblib.load('decision_tree_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')

models = {
    'DecisionTree': dt_model,
    'LogisticRegression': lr_model
}

datasets = {
    'Train': (X_train, y_train),
    'Test': (X_test, y_test),
    'Full': (X, y)
}

# 获取类别信息
classes = np.unique(y)
n_classes = len(classes)

# 判断是否为二分类
is_binary = (n_classes == 2)

# ---------------------------
# 2. 评估指标函数
# ---------------------------
def evaluate_metrics(y_true, y_pred, y_score, model_name, dataset_name):
    # 确定 average 参数
    if is_binary:
        avg = 'binary'
    else:
        avg = 'weighted'  # 或 'macro'，根据需求调整

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    # AUC 计算
    if is_binary:
        try:
            auc_val = roc_auc_score(y_true, y_score)
        except:
            auc_val = float('nan')
    else:
        # 多分类：使用 OvR 的 y_score（需概率）
        try:
            auc_val = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        except:
            auc_val = float('nan')

    print(f"{model_name} - {dataset_name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  AUC      : {auc_val:.4f}" if not np.isnan(auc_val) else "  AUC      : N/A")
    print()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc_val
    }

# ---------------------------
# 3. 遍历模型和数据集进行评估
# ---------------------------
results = {}

for model_name, model in models.items():
    results[model_name] = {}
    for dataset_name, (X_set, y_set) in datasets.items():
        y_pred = model.predict(X_set)

        # 获取预测概率（用于 ROC）
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_set)
        else:
            # 决策树应该有 predict_proba，但以防万一
            y_proba = None

        # 对于二分类，取正类概率；多分类保留全部
        if y_proba is not None:
            if is_binary:
                y_score = y_proba[:, 1]  # 假设第1列为正类
            else:
                y_score = y_proba  # shape (n_samples, n_classes)
        else:
            y_score = y_pred  # 无法计算 ROC（不推荐）

        results[model_name][dataset_name] = evaluate_metrics(
            y_set, y_pred, y_score, model_name, dataset_name
        )

# ---------------------------
# 4. 绘制 ROC 曲线（仅在 Test 集上绘制，避免过拟合幻觉）
# ---------------------------
plt.figure(figsize=(10, 8))

colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
linestyles = ['-', '--']

for idx, (model_name, model) in enumerate(models.items()):
    X_set, y_set = X_test, y_test

    if not hasattr(model, "predict_proba"):
        print(f"{model_name} 无 predict_proba，跳过 ROC 绘制。")
        continue

    y_proba = model.predict_proba(X_set)

    if is_binary:
        fpr, tpr, _ = roc_curve(y_set, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx],
                 lw=2, linestyle=linestyles[idx],
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
    else:
        # 多分类：One-vs-Rest
        y_set_bin = label_binarize(y_set, classes=classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_set_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算宏平均 ROC AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        # breakpoint()
        plt.plot(all_fpr, mean_tpr, color=colors[idx],
                 lw=2, linestyle=linestyles[idx],
                 label=f'{model_name} (macro AUC = {macro_auc:.4f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves on Test Set')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)
plt.show()

print("ROC 曲线已保存为 'roc_curves.png'")