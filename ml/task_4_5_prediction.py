import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# 复用你现有的模块
from model import train_algo_wrapper

def plot_confusion_matrix_custom(y_true, y_pred, model_name, dataset_name, output_dir='../images'):
    """
    生成并保存混淆矩阵热力图
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # 使用 Blues 配色，清爽且专业
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name}_{dataset_name}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   [Saved] Confusion Matrix ({dataset_name}) saved to {save_path}")

def visualize_decision_boundary(model_name, X, y, output_dir='../images'):
    """
    优化版决策边界可视化：
    1. 使用离散配色方案适应多分类 (7 Classes)。
    2. 训练数据和绘图数据分离，避免散点遮挡背景。
    """
    print(f"\n--- Visualizing Decision Boundary for {model_name} (2D Projection) ---")
    
    # 1. 选择两个最具代表性的特征
    feature_cols = ["Elevation", "Horizontal_Distance_To_Roadways"]
    
    if isinstance(X, pd.DataFrame):
        try:
            X_2d = X[feature_cols].values
            feature_names = feature_cols
        except KeyError:
            print(f"Warning: Specific features not found. Using first 2 columns.")
            X_2d = X.iloc[:, :2].values
            feature_names = [X.columns[0], X.columns[1]]
    else:
        X_2d = X[:, :2]
        feature_names = ["Feature 0", "Feature 1"]

    # 2. 智能采样策略
    # (A) 训练采样：样本要足够多，以画出准确的边界形状
    train_size = min(20000, len(X_2d))
    indices = np.arange(len(X_2d))
    train_idx = np.random.choice(indices, train_size, replace=False)
    
    X_train_2d = X_2d[train_idx]
    y_train_2d = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]

    # (B) 绘图采样：样本要少，避免变成“黑斑”
    plot_size = min(500, len(train_idx)) # 只画 500 个点，看起来最干净
    plot_idx = np.random.choice(train_idx, plot_size, replace=False)
    
    X_plot = X_2d[plot_idx]
    y_plot = y.iloc[plot_idx] if isinstance(y, pd.Series) else y[plot_idx]

    # 3. 重新训练 2D 代理模型
    clf_2d = train_algo_wrapper(model_name)
    clf_2d.fit(X_train_2d, y_train_2d)

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 配色方案：Pastel1 用于背景区域（淡），Set1 用于前景点（深）
    cmap_boundary = plt.get_cmap("Pastel1")
    cmap_points = plt.get_cmap("Set1")

    # 绘制背景区域
    DecisionBoundaryDisplay.from_estimator(
        clf_2d,
        X_train_2d,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
        cmap=cmap_boundary,
        alpha=0.6,
        ax=ax
    )
    
    # 绘制散点 (加白边 edgecolors='white' 让点更清晰)
    scatter = ax.scatter(
        X_plot[:, 0], 
        X_plot[:, 1], 
        c=y_plot, 
        cmap=cmap_points, 
        edgecolors="white", 
        linewidth=0.8,
        s=40, 
        alpha=0.9
    )
    
    try:
        legend1 = ax.legend(*scatter.legend_elements(), title="Cover Type", loc="upper right")
        ax.add_artist(legend1)
    except:
        pass

    plt.title(f"Decision Boundary: {model_name}\n(Features: {feature_names[0]} vs {feature_names[1]})")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    
    filename = f"decision_boundary_{model_name}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   [Saved] Optimized Decision Boundary plot saved to {save_path}")


def run_prediction_tasks(data_path='../data/covtype_processed.csv'):
    # 0. Setup
    images_dir = '../images'
    os.makedirs(images_dir, exist_ok=True)
    
    # 1. 加载数据
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 2. 数据切分 (Task 4.3)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 定义模型列表 (Task 4.2)
    models_to_run = ['DecisionTree', 'LogisticRegression']
    
    results_summary = []

    for model_name in models_to_run:
        print(f"\n" + "="*60)
        print(f"Processing Model: {model_name}")
        print("="*60)
        
        # --- Task 4.4: Train ---
        print("Training...")
        clf = train_algo_wrapper(model_name)
        clf.fit(X_train, y_train)
        
        # --- Task 4.5 & 4.6 & 5.1: Predict & Evaluate ---
        # 增加 'Full' 也就是整个数据集 (X, y)
        datasets = {
            'Train': (X_train, y_train),
            'Test':  (X_test, y_test),
            'Full':  (X, y)  # <--- Added Entire Set
        }
        
        # 临时存储该模型的指标，用于最后汇总
        model_metrics = {'Model': model_name}

        for ds_name, (X_set, y_set) in datasets.items():
            print(f"\n--- Evaluating on {ds_name} set ---")
            y_pred = clf.predict(X_set)
            
            # 计算 Accuracy
            acc = accuracy_score(y_set, y_pred)
            print(f"   -> Accuracy: {acc:.4f}")
            
            # 生成混淆矩阵 (Requirement: "generating confusion matrices ... individually")
            plot_confusion_matrix_custom(y_set, y_pred, model_name, ds_name, images_dir)
            
            # 打印分类报告
            print(f"   [Classification Report - {ds_name} Set]")
            report_dict = classification_report(y_set, y_pred, output_dict=True)
            print(classification_report(y_set, y_pred))
            
            # 收集 Test 集详细数据用于 Task 5.4 的对比表
            if ds_name == 'Test':
                model_metrics['Accuracy (Test)'] = acc
                model_metrics['Precision (Macro)'] = report_dict['macro avg']['precision']
                model_metrics['Recall (Macro)'] = report_dict['macro avg']['recall']
                model_metrics['F1 (Macro)'] = report_dict['macro avg']['f1-score']
            elif ds_name == 'Full':
                model_metrics['Accuracy (Full)'] = acc
        
        results_summary.append(model_metrics)

        # --- Task 4.3 / Deliverable 3: Visualization (Decision Boundary) ---
        try:
            visualize_decision_boundary(model_name, X_train, y_train, images_dir)
        except Exception as e:
            print(f"   [Warning] Could not plot decision boundary: {e}")

    # --- Task 5.4: Comparison Summary ---
    print("\n" + "="*60)
    print(">>> Final Model Comparison Summary")
    print("="*60)
    summary_df = pd.DataFrame(results_summary)
    
    # 调整列顺序，把重要的放在前面
    cols = ['Model', 'Accuracy (Test)', 'Accuracy (Full)', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)']
    # 防止有些列没生成（防守性编程）
    cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df[cols]
    
    print(summary_df)
    summary_df.to_csv(os.path.join(images_dir, 'model_comparison_metrics.csv'), index=False)
    print(f"\nMetrics saved to {os.path.join(images_dir, 'model_comparison_metrics.csv')}")

if __name__ == "__main__":
    run_prediction_tasks()