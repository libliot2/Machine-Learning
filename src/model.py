import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

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

# 假设前54列为特征，最后一列为标签
X = data.iloc[:, :-1]  # 前54列
y = data.iloc[:, -1]   # 最后一列

# 2. 划分训练集（70%）和测试集（30%），设置 random_state 保证可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # 如果 y 是分类标签，stratify 有助于保持分布
)

# 3. 初始化并训练两个模型

# 决策树
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 逻辑回归（注意：逻辑回归默认只支持数值型输入，且对于多分类支持良好）
# 如果标签是多分类且非连续整数，可自动处理；若为字符串，LabelEncoder 可能需要
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 4. 在训练集、测试集、完整数据集上评估模型准确率
models = {
    'DecisionTree': dt_model,
    'LogisticRegression': lr_model
}

datasets = {
    'Train': (X_train, y_train),
    'Test': (X_test, y_test),
    'Full': (X, y)
}

for model_name, model in models.items():
    print(f"\n=== {model_name} ===")
    for dataset_name, (X_set, y_set) in datasets.items():
        y_pred = model.predict(X_set)
        acc = accuracy_score(y_set, y_pred)
        print(f"{dataset_name} Accuracy: {acc:.4f}")

# 5. 保存模型（使用 joblib，适合保存 sklearn 模型）
joblib.dump(dt_model, 'decision_tree_model.pkl')
joblib.dump(lr_model, 'logistic_regression_model.pkl')

print("\n模型已保存为 'decision_tree_model.pkl' 和 'logistic_regression_model.pkl'")