import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 加载数据 (假设 covtype_processed.csv 已存在)
# ==========================================
# 为了演示"原始尺度差异"，我们其实需要未标准化的数据。
# 如果你只有处理后的数据，请改回加载原始数据。这里假设我们要论证"为什么需要标准化"，
# 所以我们模拟加载原始数据的前10列。

print("正在加载数据用于 EDA...")
# 重新定义列名 (为了准确性)
cols_num = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
]

# 为了内存效率，我们先只读取这些列，加上 Soil_Types
# Soil Types 是最后 40 列 (索引 14 到 53)
# 这里为了代码简洁，我们直接读取全部，然后切片
try:
    df = pd.read_csv('covtype_processed.csv') 
    # 注意：如果读取的是已经 Standardized 的数据，Boxplot 就看不出差异了！
    # 如果 covtype_processed.csv 是已经缩放过的，我们需要重新加载原始数据来画图1。
    # 为了保险，这里演示重新加载原始数据流（如果本地有 processed，请自行确认是否已缩放）
    print("注意：为了展示尺度差异，正在尝试加载原始数据(covtype.data.gz)...")
    
    # 完整列名列表
    all_cols = cols_num + [f"Wilderness_Area_{i}" for i in range(1,5)] + \
               [f"Soil_Type_{i}" for i in range(1,41)] + ["Cover_Type"]
    
    df_raw = pd.read_csv('covtype.data.gz', header=None, names=all_cols, compression='gzip')

except FileNotFoundError:
    print("未找到原始数据，尝试使用当前环境中的 dataframe...")
    # 这里的 df_raw 应该是你内存中未标准化的数据
    df_raw = df # 如果没有原始文件，就用当前的（如果是已缩放的，Boxplot意义会变）

# ==========================================
# 2. Visualization 1: Feature Scaling (Boxplot)
# ==========================================
print("生成特征尺度箱线图 (Boxplot)...")

plt.figure(figsize=(14, 8))

# 使用对数刻度? 不，原始刻度更能展示差异。
# 我们画前 10 个连续特征
sns.boxplot(data=df_raw[cols_num], orient='h', palette="Set2")

plt.title('Distribution of Continuous Features (Raw Scale)\nDemonstrating the need for StandardScaler', fontsize=15)
plt.xlabel('Value Range (Raw Units)')
plt.ylabel('Features')
plt.grid(True, alpha=0.3)

# 保存
plt.tight_layout()
plt.savefig('eda_feature_scaling_boxplot.png')
print("已保存: eda_feature_scaling_boxplot.png")
plt.close()

# ==========================================
# 3. Visualization 2: Correlation Heatmap
# ==========================================
print("生成相关性热力图 (Heatmap)...")

plt.figure(figsize=(10, 8))
corr_matrix = df_raw[cols_num].corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Continuous Features', fontsize=15)

plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
print("已保存: eda_correlation_heatmap.png")
plt.close()

# ==========================================
# 4. Visualization 3: Soil Type Sparsity (Bar Chart)
# ==========================================
print("生成 Soil Type 稀疏性分析图 (Bar Chart)...")

# 提取 Soil Type 列
soil_cols = [col for col in df_raw.columns if "Soil_Type" in col]
# 计算每一列中 "1" 的总数 (即该类土壤出现的次数)
soil_counts = df_raw[soil_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(16, 8))
sns.barplot(x=soil_counts.index, y=soil_counts.values, palette="magma")

# 美化 x 轴标签 (太长了，旋转一下，或者只显示编号)
# 提取编号： "Soil_Type_10" -> "10"
xtick_labels = [label.split('_')[-1] for label in soil_counts.index]
plt.xticks(ticks=range(len(soil_counts)), labels=xtick_labels, rotation=0, fontsize=9)

plt.title('Frequency of Soil Types (Checking for Sparsity)', fontsize=15)
plt.xlabel('Soil Type Index', fontsize=12)
plt.ylabel('Count of Samples', fontsize=12)
plt.yscale('log') # 关键！因为差异太大，用对数坐标轴能看清那些极少的类
plt.grid(axis='y', alpha=0.3)

# 添加说明
plt.text(0.5, 0.9, 'Note: Y-axis is Log Scale due to high imbalance', 
         transform=plt.gca().transAxes, ha='center', fontsize=12, color='red')

plt.tight_layout()
plt.savefig('eda_soil_sparsity.png')
print("已保存: eda_soil_sparsity.png")
plt.close()

print("所有 EDA 图表生成完毕！")