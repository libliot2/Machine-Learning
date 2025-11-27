import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import convert_to_float_if_needed

def cluster_and_evaluate(
    df: pd.DataFrame,
    sample_size: int = 51200,
    random_state: int = 42,
    n_clusters: int = 7,
    output_path: str = '../images',
    cluster_algorithms: list = ['Kmeans', 'Hierarchical']
):
    df = convert_to_float_if_needed(df[1:])
    Sample_Size = min(sample_size, len(df))
    df = df.sample(n=Sample_Size, random_state=random_state)

    features = df.iloc[:, :-1].values 
    true_labels = df.iloc[:, -1].values


    print(f"Shape of Dataset: {df.shape}")
    print(f"Shape of Feature Matrix: {features.shape}")

    print("\nExecuting PCA for Visualization...")
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    viz_df = pd.DataFrame(features_2d, columns=['PC1', 'PC2'])
    fig, axes = plt.subplots(1, len(cluster_algorithms), figsize=(14, 3 * len(cluster_algorithms)))

    for algo_name in cluster_algorithms:
        viz_df[f'{algo_name}_Cluster'] = cluster_algo_wrraper(
            algo_name=algo_name,
            features=features
        )

    for i in range(len(cluster_algorithms)):
        sns.scatterplot(data=viz_df, x='PC1', y='PC2', hue=f'{cluster_algorithms[i]}_Cluster', palette='tab10', ax=axes[i], legend='full')
        axes[i].set_title(f'{algo_name} Clustering (k={n_clusters}) - PCA Visualization')
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "clustering_visualization.png"))
    plt.show()

    print("\nClustering and Visualization Completed！")

def compute_clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str
):
    silhouette_avg = silhouette_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)

    print(f"\n{algorithm_name} Clustering Metrics:")
    print(f"  - Silhouette Score: {silhouette_avg:.4f}")
    print(f"  - Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    print(f"  - Davies-Bouldin Index: {davies_bouldin:.4f}")

def cluster_algo_wrraper(
    algo_name: str = "Kmeans",
    features: np.ndarray = None,
):
    if algo_name == "Kmeans":
        print("Excecuting K-means ...")
        return kmeans_clustering(features)
    elif algo_name == "Hierarchical":
        print("Excecuting Hierarchical ...")
        return hierarchical_clustering(features)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algo_name}")

def kmeans_clustering(
    features: np.ndarray,
    n_clusters: int = 7,
) -> np.ndarray:
    print(f"\nExecuting K-means Cluster (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features)
    compute_clustering_metrics(features, kmeans_labels, "K-means")
    return kmeans_labels
    

def hierarchical_clustering(
    features: np.ndarray,
    n_clusters: int = 7,
) -> np.ndarray:
    print(f"Executing Hierarchical (n_clusters={n_clusters})...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(features)
    compute_clustering_metrics(features, hierarchical_labels, "Hierarchical")
    return hierarchical_labels