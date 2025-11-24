import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 读取数据 ---
# 假设您的数据文件名为 'your_dataset.csv'，请将其替换为实际文件名
print("正在读取数据...")
df = pd.read_csv('covtype_processed.csv')

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

Sample_Size = 51200
random_state = 42
df = convert_to_float_if_needed(df[1:])
Sample_Size = min(Sample_Size, len(df))
df = df.sample(n=Sample_Size, random_state=random_state)

# 分离特征和标签
# 假设列名是标准的，即 '0', '1', ..., '53' 是特征，'54' 是标签
# 如果列名不同，请根据实际情况调整
features = df.iloc[:, :-1].values  # 前54个特征
true_labels = df.iloc[:, -1].values # 最后一列作为标签（如果需要，可用于后续比较）

print(f"数据集形状: {df.shape}")
print(f"特征矩阵形状: {features.shape}")

# --- 2. 进行K-means和层次聚类 ---
# 为了可视化效果，我们假设聚类数量为3。您可以根据需要调整。
n_clusters = 7

print(f"\n正在进行K-means聚类 (k={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(features)

print(f"正在进行层次聚类 (n_clusters={n_clusters})...")
hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
hierarchical_labels = hierarchical.fit_predict(features)

# --- 3. 评估聚类结果 ---
print("\n--- 聚类结果评估 ---")

for name, labels in [("K-means", kmeans_labels), ("Hierarchical", hierarchical_labels)]:
    print(f"\n--- {name} ---")
    
    # Silhouette Score (轮廓系数)
    sil_score = silhouette_score(features, labels)
    print(f"轮廓系数 (Silhouette Score): {sil_score:.4f}")
    
    # Calinski-Harabasz Score (Calinski-Harabasz离散度)
    # 该指标需要至少2个簇，且当簇数为样本数时未定义
    if n_clusters > 1 and n_clusters < len(features):
        ch_score = calinski_harabasz_score(features, labels)
        print(f"Calinski-Harabasz离散度 (CH Score): {ch_score:.4f}")
    else:
        print("Calinski-Harabasz离散度: 不适用 (簇数需在1和样本数之间)")
    
    # Davies-Bouldin Score (Davies-Bouldin相似度)
    db_score = davies_bouldin_score(features, labels)
    print(f"Davies-Bouldin相似度 (DB Score): {db_score:.4f}")


# --- 4. 可视化聚类结果 ---
# 由于原始数据是54维的，无法直接可视化。
# 我们使用PCA将数据降维到2维进行散点图绘制。
print("\n正在进行PCA降维以进行可视化...")
pca = PCA(n_components=2, random_state=42)
features_2d = pca.fit_transform(features)

# 创建一个DataFrame用于seaborn绘图
viz_df = pd.DataFrame(features_2d, columns=['PC1', 'PC2'])
viz_df['KMeans_Cluster'] = kmeans_labels
viz_df['Hierarchical_Cluster'] = hierarchical_labels

# 绘制子图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-means可视化
sns.scatterplot(data=viz_df, x='PC1', y='PC2', hue='KMeans_Cluster', palette='tab10', ax=axes[0], legend='full')
axes[0].set_title(f'K-means Clustering (k={n_clusters}) - PCA Visualization')
axes[0].grid(True)

# Hierarchical可视化
sns.scatterplot(data=viz_df, x='PC1', y='PC2', hue='Hierarchical_Cluster', palette='tab10', ax=axes[1], legend='full')
axes[1].set_title(f'Hierarchical Clustering (n_clusters={n_clusters}) - PCA Visualization')
axes[1].grid(True)

# 调整布局，防止重叠
plt.tight_layout()
plt.savefig("clustering_visualization.png")
plt.show()

print("\n聚类和可视化完成！")