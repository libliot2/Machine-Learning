import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 定义列名 & 加载数据
# ==========================================
print(">>> Step 1: Loading Data...")

# 手动定义列名 (Covertype Dataset通常没有表头)
cols = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
]

# 添加 4 个 Wilderness_Area 列
for i in range(1, 5):
    cols.append(f"Wilderness_Area_{i}")

# 添加 40 个 Soil_Type 列
for i in range(1, 41):
    cols.append(f"Soil_Type_{i}")

# 添加目标列
cols.append("Cover_Type")

# 加载数据 (假设是 covtype.data.gz，如果解压了请改文件名)
try:
    # header=None 表示文件中没有列名行
    df = pd.read_csv('covtype.data.gz', header=None, names=cols, compression='gzip')
except FileNotFoundError:
    # 备用：如果用户已经解压或是 .csv 格式
    print("未找到 .gz 文件，尝试加载 .csv ...")
    df = pd.read_csv('covtype.csv', header=None, names=cols)

print(f"数据加载完成。形状: {df.shape}")

# ==========================================
# 2. 数据预处理 (Mandatory Task 1)
# ==========================================
print("\n>>> Step 2: Preprocessing (Task 1)...")

# 2.1 检查缺失值
missing = df.isnull().sum().sum()
print(f"缺失值总数: {missing}") 

# 2.2 检查非数值列
non_numeric = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric) == 0:
    print("所有列均为数值型，无需额外编码。")
else:
    print(f"发现非数值列: {non_numeric}")

# 2.3 特征标准化 (Standardization)
# 策略：只缩放前 10 个连续特征，保留二值特征 (One-hot) 不变
continuous_features = cols[:10] 
binary_features = cols[10:-1]   
target_feature = "Cover_Type"   

print(f"正在对 {len(continuous_features)} 个连续特征进行标准化...")

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[continuous_features] = scaler.fit_transform(df[continuous_features])

print("标准化完成。")

# ==========================================
# 3. 数据分布可视化 (Mandatory Task 2 Part A)
# ==========================================
print("\n>>> Step 3: Visualization & Analysis...")

# 3.1 统计类别分布
target_col = 'Cover_Type'
class_counts = df_scaled[target_col].value_counts().sort_index()

print("Class Distribution (类别分布):")
print(class_counts)

# 3.2 绘图并保存 (适配无头服务器环境)
plt.figure(figsize=(10, 6))
sns.barplot(
    x=class_counts.index, 
    y=class_counts.values, 
    hue=class_counts.index, 
    palette='viridis', 
    legend=False
)
plt.title('Distribution of Forest Cover Types')
plt.xlabel('Cover Type')
plt.ylabel('Count')

# 保存图片
output_img = "class_distribution.png"
plt.savefig(output_img)
print(f"分布图已保存为: {output_img}")
plt.close()

# ==========================================
# 4. 保存处理后的数据
# ==========================================
print("\n>>> Step 4: Saving Processed Data...")
output_csv = 'covtype_processed.csv'
# index=False 避免保存时多出一列索引
df_scaled.to_csv(output_csv, index=False)
print(f"预处理后的数据已保存为: {output_csv}")
print("Done.")