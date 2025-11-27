import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing import data_preprocessing
from visualization import eda_visual, tsne_visual, pca_visual
from cluster_algo import cluster_and_evaluate
from model import model_train
from model_eval import eval_models
from feature_engineering import (
    combine_distance_features,
    aspect_slope_interaction,
    hillshade_stats,
    encode_categorical_areas,
    elevation_interactions,
)

def main(
    raw_dataset_path: str = '../data/covtype.data.gz',
    dataset_path: str = '../data/covtype_processed.csv',
    is_preprocess: bool = False,
):
    if is_preprocess:
        data = data_preprocessing(
            file_name=raw_dataset_path,
        )
    else:
        data = pd.read_csv(dataset_path)
    eda_visual()
    tsne_visual(data)

    cluster_algos = ['Kmeans', 'Hierarchical']
    cluster_and_evaluate(
        data,
        sample_size=5120,
        cluster_algorithms=cluster_algos,
    )

    X = data.iloc[:, :-1] 
    y = data.iloc[:, -1]   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # 如果 y 是分类标签，stratify 有助于保持分布
    )

    datasets = {
        'Train': (X_train, y_train),
        'Test': (X_test, y_test),
        'Full': (X, y)
    }

    models_name = ['DecisionTree', 'LogisticRegression', 'RandomForest', 'NeuralNetwork']

    for model_name in models_name:
        model_train(
            model_name=model_name,
            datasets=datasets,
        )
    
    model_files = [f"../model/{name}_model.pkl" for name in models_name]
    eval_models(
        datasets=datasets,
        models_path=model_files,
    )


if __name__ == "__main__":
    dataset_path = '../data/covtype.data.gz'
    main(dataset_path)