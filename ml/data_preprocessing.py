import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

def data_preprocessing(
    file_name: str = 'covtype.data.gz',
    image_output_path: str = '../images',
    data_output_path: str = '../data'
):
    print(">>> Step 1: Loading Data...")

    cols = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
    ]

    for i in range(1, 5):
        cols.append(f"Wilderness_Area_{i}")

    for i in range(1, 41):
        cols.append(f"Soil_Type_{i}")

    cols.append("Cover_Type")

    df = pd.read_csv(file_name, header=None, names=cols, compression='gzip')

    print(f"Data Load Finished. Shape: {df.shape}")

    print("\n>>> Step 2: Preprocessing (Task 1)...")

    missing = df.isnull().sum().sum()
    print(f"Missing Numbers Num: {missing}") 

    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) == 0:
        print("All columns are numeric.")
    else:
        print(f"Find Non-numeric Column: {non_numeric}")

    continuous_features = cols[:10] 
    binary_features = cols[10:-1]   
    target_feature = "Cover_Type"   

    print(f"Normalizing {len(continuous_features)} ...")

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[continuous_features] = scaler.fit_transform(df[continuous_features])


    print("\n>>> Step 3: Visualization & Analysis...")

    target_col = 'Cover_Type'
    class_counts = df_scaled[target_col].value_counts().sort_index()

    print("Class Distribution:")
    print(class_counts)

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

    output_img = os.path.join(image_output_path, "class_distribution.png")
    plt.savefig(output_img)
    print(f"Image Saved As: {output_img}")
    plt.close()

    print("\n>>> Step 4: Saving Processed Data...")
    output_csv = os.path.join(data_output_path, 'covtype_processed.csv')

    df_scaled.to_csv(output_csv, index=False)
    print(f"Data Preprocessed is Save as: {output_csv}")
    print("Done.")
    return df_scaled