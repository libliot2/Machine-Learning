import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# ---- 用户配置区 ----
csv_file = 'covtype_processed.csv'      # 请替换为你的实际文件路径
Sample_Size = 5120            # 采样数量（建议不超过 10000，t-SNE 较慢）
random_state = 42

# ---- 1. 读取、检查并转换数据类型 ----
print("正在读取数据...")
df = pd.read_csv(csv_file, header=None)
print(f"原始数据形状: {df.shape}")

# 检查并转换数据类型：将所有非浮点数列转为浮点数
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

df = convert_to_float_if_needed(df[1:])

# ---- 2. 采样数据 ----
# 确保采样数量不超过总样本数
Sample_Size = min(Sample_Size, len(df))
sampled_df = df.sample(n=Sample_Size, random_state=random_state)

X = sampled_df.iloc[:, :54].values
y = sampled_df.iloc[:, 54].values

print(f"采样后特征矩阵形状: {X.shape}, 标签形状: {y.shape}")
print(f"特征数据类型: {X.dtype}, 标签数据类型: {y.dtype}")

# ---- 3. t-SNE 降维 ----
print("正在执行 t-SNE 降维（这可能需要几分钟）...")
tsne = TSNE(
    n_components=2,
    random_state=random_state,
    perplexity=min(30, len(X) - 1),  # perplexity 不能大于 n_samples - 1
    max_iter=1000,
    n_jobs=-1  # 使用所有 CPU 加速（若支持）
)
X_tsne = tsne.fit_transform(X)

# ---- 4. 绘图 ----
print("正在绘图...")
unique_labels = np.unique(y)
n_labels = len(unique_labels)

# 自动选择 colormap
if n_labels <= 10:
    cmap = 'tab10'
elif n_labels <= 20:
    cmap = 'tab20'
else:
    cmap = 'nipy_spectral'  # 支持更多类别

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, s=8, alpha=0.7)
plt.title(f't-SNE Visualization (Sample Size = {Sample_Size})', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# 若类别不多，添加图例
if n_labels <= 20:
    # 为标签创建图例，处理可能的浮点数标签
    unique_labels_rounded = [int(label) if label.is_integer() else label for label in unique_labels]
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=plt.cm.get_cmap(cmap)(i / max(1, n_labels - 1)), 
                                  markersize=6, label=str(unique_labels_rounded[i])) 
                       for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('tsne_plot.png', dpi=300, bbox_inches='tight')
print("✅ 图像已保存为 'tsne_plot.png'")

# 可选：如果在支持显示的环境中，取消下一行注释
# plt.show()