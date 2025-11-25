import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# 1. 读取数据
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

X = data.iloc[:, :-1].values  # 前54列为特征
y = data.iloc[:, -1].values   # 最后一列为标签

# 2. 划分训练集（70%）和测试集（30%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify 保持标签分布一致（如果标签类别不平衡）
)

# 3. 定义模型
models = {
    "SVM": SVC(kernel='rbf', random_state=42, max_iter=10000),  # RBF核，限制最大迭代次数
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# 4. 训练模型并评估
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # 预测与评估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    all_pred = model.predict(X)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    all_acc = accuracy_score(y, all_pred)
    
    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'full_accuracy': all_acc
    }
    
    # 保存模型
    joblib.dump(model, f"{name}_model.joblib")
    print(f"{name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Full Acc: {all_acc:.4f}")

# 5. 打印汇总结果
print("\n=== Final Results ===")
for name, scores in results.items():
    print(f"{name}:")
    print(f"  Train Accuracy: {scores['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {scores['test_accuracy']:.4f}")
    print(f"  Full Accuracy:  {scores['full_accuracy']:.4f}")