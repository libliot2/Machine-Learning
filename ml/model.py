import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os



def model_train(
    model_name: str,
    output_path: str = '../model',
    datasets: dict = None,
):
    model = train_algo_wrapper(model_name)
    X_train, y_train = datasets['Train']
    model.fit(X_train, y_train)
    model_file = f"{model_name}_model.pkl"
    joblib.dump(model, os.path.join(output_path, model_file))
    print(f"Model {model_name} saved to {model_file}")


def train_algo_wrapper(
    algo_name: str,
):
    if algo_name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=42)
    elif algo_name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, random_state=42)
    elif algo_name == 'SVM':
        return SVC(random_state=42)
    elif algo_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif algo_name == 'NeuralNetwork':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
