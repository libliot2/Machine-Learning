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
import os

colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'yellow']
linestyles = ['-', '--']

def eval_models(
    datasets: dict,
    models_path: list,
    images_output_dir: str = '../images',
):
    models = {}
    for model_path in models_path:
        model_name = os.path.basename(model_path)
        models[model_name] = joblib.load(model_path)

    classes = np.unique(datasets['Train'][1])
    n_classes = len(classes)
    is_binary = (n_classes == 2)
    
    results = {}
    for model_name in models.keys():
        results[model_name] = {}
        model = models[model_name]
        for dataset_name, (X_set, y_set) in datasets.items():
            y_pred = model.predict(X_set)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_set)
            else:
                y_proba = None

            if y_proba is not None:
                if is_binary:
                    y_score = y_proba[:, 1]
                else:
                    y_score = y_proba 
            else:
                y_score = y_pred 

            results[model_name][dataset_name] = evaluate_metrics(
                y_set, y_pred, y_score, model_name, dataset_name, is_binary,
            )

    plt.figure(figsize=(10, 8))

    for idx, (model_name, model) in enumerate(models.items()):
        (X_set, y_set) = datasets['Test']

        model = models[model_name]
        if not hasattr(model, "predict_proba"):
            print(f"{model_name} doesn't have predict_proba, Skip ROC Plot。")
            continue

        y_proba = model.predict_proba(X_set)

        if is_binary:
            fpr, tpr, _ = roc_curve(y_set, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[idx],
                    lw=2, linestyle=linestyles[idx % 2],
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
        else:
            y_set_bin = label_binarize(y_set, classes=classes)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_set_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, color=colors[idx],
                    lw=2, linestyle=linestyles[idx % 2],
                    label=f'{model_name} (macro AUC = {macro_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves on Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(images_output_dir, ('roc_curves.png')), dpi=300)
    plt.show()

    print("ROC Save as 'roc_curves.png'")



def evaluate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_score: np.ndarray, 
    model_name: str, 
    dataset_name: str,   
    is_binary: bool = True,   
):
    if is_binary:
        avg = 'binary'
    else:
        avg = 'weighted'  

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    if is_binary:
        try:
            auc_val = roc_auc_score(y_true, y_score)
        except:
            auc_val = float('nan')
    else:
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

