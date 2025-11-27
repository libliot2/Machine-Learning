import pandas as pd
import numpy as np

def combine_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 原始距离列
    hdh = 'Horizontal_Distance_To_Hydrology'
    vdh = 'Vertical_Distance_To_Hydrology'
    hdr = 'Horizontal_Distance_To_Roadways'
    hdf = 'Horizontal_Distance_To_Fire_Points'
    
    # 欧氏距离到水源
    df['Euclidean_Distance_To_Hydrology'] = np.sqrt(df[hdh]**2 + df[vdh]**2)
    
    # 所有水平距离之和或加权和
    df['Sum_Horizontal_Distances'] = df[hdh] + df[hdr] + df[hdf]
    df['Mean_Horizontal_Distances'] = (df[hdh] + df[hdr] + df[hdf]) / 3.0
    
    return df

def aspect_slope_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Aspect_Slope_Interaction'] = df['Aspect'] * df['Slope']
    df['Aspect_Sin'] = np.sin(np.radians(df['Aspect']))
    df['Aspect_Cos'] = np.cos(np.radians(df['Aspect']))
    return df

def hillshade_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hillshade_cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    df['Hillshade_Mean'] = df[hillshade_cols].mean(axis=1)
    df['Hillshade_Std'] = df[hillshade_cols].std(axis=1)
    df['Hillshade_Range'] = df[hillshade_cols].max(axis=1) - df[hillshade_cols].min(axis=1)
    return df

def encode_categorical_areas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Wilderness_Area: 列名 Wilderness_Area_1 ~ _4
    wilderness_cols = [col for col in df.columns if col.startswith('Wilderness_Area_')]
    df['Wilderness_Area'] = df[wilderness_cols].idxmax(axis=1).str.extract(r'(\d+)')[0].astype(int)
    
    # Soil_Type: Soil_Type_1 ~ _40
    soil_cols = [col for col in df.columns if col.startswith('Soil_Type_')]
    df['Soil_Type'] = df[soil_cols].idxmax(axis=1).str.extract(r'(\d+)')[0].astype(int)
    
    # 可选：保留原始 one-hot 或删除
    df = df.drop(columns=wilderness_cols + soil_cols)
    
    return df

def elevation_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Elevation vs distances
    df['Elev_Distance_Hydro_Ratio'] = df['Elevation'] / (df['Horizontal_Distance_To_Hydrology'] + 1e-6)
    df['Elev_Distance_Road_Diff'] = df['Elevation'] - df['Horizontal_Distance_To_Roadways']
    df['Elev_Distance_Fire_Sum'] = df['Elevation'] + df['Horizontal_Distance_To_Fire_Points']
    return df