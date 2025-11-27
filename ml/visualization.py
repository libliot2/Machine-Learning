import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os
from utils import convert_to_float_if_needed

def eda_visual(
    file_path: str = '../data/covtype_processed.csv',
    raw_file_name: str = '../data/covtype.data.gz',
    out_images_path: str = '../images',
):
    print("Loading Data for EDA...")
    cols_num = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
    ]

    try:
        df = pd.read_csv(file_path) 
        
        all_cols = cols_num + [f"Wilderness_Area_{i}" for i in range(1,5)] + \
                [f"Soil_Type_{i}" for i in range(1,41)] + ["Cover_Type"]
        
        df_raw = df

    except FileNotFoundError:
        print("Doesn't find original data, using existing dataframe...")

    print("Generate (Boxplot)...")

    plt.figure(figsize=(14, 8))

    sns.boxplot(data=df_raw[cols_num], orient='h', palette="Set2")

    plt.title('Distribution of Continuous Features (Raw Scale)\nDemonstrating the need for StandardScaler', fontsize=15)
    plt.xlabel('Value Range (Raw Units)')
    plt.ylabel('Features')
    plt.grid(True, alpha=0.3)

    # 保存
    plt.tight_layout()
    plt.savefig(os.path.join(out_images_path,'eda_feature_scaling_boxplot.png'))
    print("Saved as eda_feature_scaling_boxplot.png")
    plt.close()

    print("Generate (Heatmap)...")

    plt.figure(figsize=(10, 8))
    corr_matrix = df_raw[cols_num].corr()

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Continuous Features', fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(out_images_path, 'eda_correlation_heatmap.png'))
    print("Saved as eda_correlation_heatmap.png")
    plt.close()

    print("Generate Soil Type  (Bar Chart)...")

    soil_cols = [col for col in df_raw.columns if "Soil_Type" in col]
    soil_counts = df_raw[soil_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(16, 8))
    sns.barplot(x=soil_counts.index, y=soil_counts.values, palette="magma")

    xtick_labels = [label.split('_')[-1] for label in soil_counts.index]
    plt.xticks(ticks=range(len(soil_counts)), labels=xtick_labels, rotation=0, fontsize=9)

    plt.title('Frequency of Soil Types (Checking for Sparsity)', fontsize=15)
    plt.xlabel('Soil Type Index', fontsize=12)
    plt.ylabel('Count of Samples', fontsize=12)
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)

    plt.text(0.5, 0.9, 'Note: Y-axis is Log Scale due to high imbalance', 
            transform=plt.gca().transAxes, ha='center', fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(out_images_path, 'eda_soil_sparsity.png'))
    print("Saved as eda_soil_sparsity.png")
    plt.close() 

def tsne_visual(
    df: pd.DataFrame,
    sample_size: int = 5000,
    random_state: int = 42,
    output_path: str = '../images',
):
    df = convert_to_float_if_needed(df[1:])  # Skip first row if it's header or invalid
    Sample_Size = min(sample_size, len(df))
    sampled_df = df.sample(n=Sample_Size, random_state=random_state)

    X = sampled_df.iloc[:, :54].values
    y = sampled_df.iloc[:, 54].values

    print(f"Shape of Feature Matrix after Sampling: {X.shape}, Shape of Lable Matrix : {y.shape}")

    print("Executing t-SNE ...")
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=min(30, len(X) - 1),  # perplexity 不能大于 n_samples - 1
        max_iter=1000,
        n_jobs=-1  # 使用所有 CPU 加速（若支持）
    )
    X_tsne = tsne.fit_transform(X)

    print("Ploting...")
    unique_labels = np.unique(y)
    n_labels = len(unique_labels)

    if n_labels <= 10:
        cmap = 'tab10'
    elif n_labels <= 20:
        cmap = 'tab20'
    else:
        cmap = 'nipy_spectral'  

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, s=5, alpha=0.6)
    plt.title(f't-SNE Visualization (Sample Size = {Sample_Size})', fontsize=14)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    if n_labels <= 20:
        unique_labels_rounded = [int(label) if label.is_integer() else label for label in unique_labels]
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.get_cmap(cmap)(i / max(1, n_labels - 1)), 
                                    markersize=6, label=str(unique_labels_rounded[i])) 
                        for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'tsne_plot.png'), dpi=300, bbox_inches='tight')
    print("✅ Images Saved As 'tsne_plot.png'")

def pca_visual(
    df: pd.DataFrame,
    sample_size: int = 5000,
    random_state: int = 42,
    output_path: str = '../images',
):
    df = convert_to_float_if_needed(df[1:])  # Skip first row if it's header or invalid
    Sample_Size = min(sample_size, len(df))
    sampled_df = df.sample(n=Sample_Size, random_state=random_state)

    X = sampled_df.iloc[:, :54].values
    y = sampled_df.iloc[:, 54].values

    print(f"Shape of Feature Matrix after Sampling: {X.shape}, Shape of Lable Matrix : {y.shape}")

    print("Executing PCA 降维...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    print(f"PCA explained_variance_ratio: {pca.explained_variance_ratio_}")

    print("Ploting...")
    unique_labels = np.unique(y)
    n_labels = len(unique_labels)

    if n_labels <= 10:
        cmap = 'tab10'
    elif n_labels <= 20:
        cmap = 'tab20'
    else:
        cmap = 'nipy_spectral'

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, s=8, alpha=0.7)
    plt.title(f'PCA Visualization (Sample Size = {Sample_Size})', fontsize=14)
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

    # 若类别不多，添加图例
    if n_labels <= 20:
        unique_labels_rounded = [int(label) if label.is_integer() else label for label in unique_labels]
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=plt.cm.get_cmap(cmap)(i / max(1, n_labels - 1)),
                    markersize=6, label=str(unique_labels_rounded[i]))
            for i in range(len(unique_labels))
        ]
        plt.legend(handles=legend_elements, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'pca_plot.png'), dpi=300, bbox_inches='tight')
    print("✅ Images Saved As 'pca_plot.png'")