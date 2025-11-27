import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# 确保输出目录存在
os.makedirs('../images', exist_ok=True)

def cluster_and_evaluate(
    df: pd.DataFrame,
    sample_size: int = 100000, 
    random_state: int = 42,
    n_clusters: int = 7,      
    output_path: str = '../images',
    cluster_algorithms: list = ['Kmeans', 'Hierarchical', 'GMM'] 
):
    """
    执行聚类、评估并可视化结果。
    """
    
    # --- 1. 数据准备 (分级采样) ---
    VIZ_LIMIT = 15000 
    
    # 可视化子集
    viz_n = min(VIZ_LIMIT, len(df))
    print(f"Sampling {viz_n} instances for Visualization & Hierarchical Clustering...")
    df_viz = df.sample(n=viz_n, random_state=random_state)
    viz_features = df_viz.iloc[:, :-1].values
    
    # 训练子集
    train_n = min(sample_size, len(df))
    print(f"Sampling {train_n} instances for Model Training (K-Means/GMM)...")
    df_train = df.sample(n=train_n, random_state=random_state)
    train_features = df_train.iloc[:, :-1].values

    # --- 2. PCA 降维 ---
    print("\nExecuting PCA for Visualization (on viz subset)...")
    pca = PCA(n_components=2, random_state=random_state)
    features_2d = pca.fit_transform(viz_features)
    viz_df = pd.DataFrame(features_2d, columns=['PC1', 'PC2'])

    # --- 3. 聚类循环 ---
    num_algos = len(cluster_algorithms)
    fig, axes = plt.subplots(1, num_algos, figsize=(6 * num_algos, 5))
    if num_algos == 1: axes = [axes]

    # 直接开始循环，不强制锁死线程，依赖外部环境变量
    for i, algo_name in enumerate(cluster_algorithms):
        print(f"\n>>> Running Algorithm: {algo_name}")
        
        if algo_name == 'Hierarchical':
            if train_n > VIZ_LIMIT:
                print(f"   (Restricted to {viz_n} samples due to memory constraints)")
            labels_for_viz = hierarchical_clustering(viz_features, n_clusters)
        else:
            labels_for_viz = train_predict_wrapper(
                algo_name, train_features, viz_features, n_clusters, random_state
            )
        
        viz_df[f'{algo_name}_Cluster'] = labels_for_viz

        # --- 4. 可视化 ---
        sns.scatterplot(
            data=viz_df, 
            x='PC1', y='PC2', 
            hue=f'{algo_name}_Cluster', 
            palette='tab10', 
            ax=axes[i], 
            legend='full',
            s=10, alpha=0.6
        )
        axes[i].set_title(f'{algo_name} (k={n_clusters})')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_path, "clustering_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nVisualization saved to: {save_path}")
    print("\nAll Clustering Tasks Completed!")

def train_predict_wrapper(algo_name, train_features, viz_features, n_clusters, random_state):
    """
    通用包装器：在大样本上训练，在小样本上预测
    """
    model = None
    labels_train = None
    
    if algo_name == "Kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        model.fit(train_features)
        labels_train = model.labels_
        
    elif algo_name == "GMM":
        model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=random_state, max_iter=100)
        labels_train = model.fit_predict(train_features)
    
    compute_clustering_metrics(train_features, labels_train, algo_name)
    
    labels_viz = model.predict(viz_features)
    return labels_viz

def hierarchical_clustering(features, n_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(features)
    compute_clustering_metrics(features, labels, "Hierarchical")
    return labels

def compute_clustering_metrics(features, labels, algorithm_name):
    """
    计算三个核心内部评估指标
    """
    print(f"   Calculating metrics for {algorithm_name} (Samples: {len(features)})...")
    
    # 轮廓系数: 采样计算以提速
    sil_score = silhouette_score(features, labels, metric='euclidean', sample_size=5000)
    
    ch_score = calinski_harabasz_score(features, labels)
    db_score = davies_bouldin_score(features, labels)

    print(f"   [{algorithm_name}] -> Silhouette: {sil_score:.4f} | CH: {ch_score:.1f} | DB: {db_score:.4f}")

def cluster_algo_wrapper(algo_name, features, n_clusters, random_state):
    if algo_name == 'Hierarchical':
        return hierarchical_clustering(features, n_clusters)
    else:
        return train_predict_wrapper(algo_name, features, features, n_clusters, random_state)