import pandas as pd
import numpy as np

def convert_to_float_if_needed(df):
    """
    Try to Convert DataFrame columns to float32 if they are not already floats.
    """
    for col in df.columns:
        if not pd.api.types.is_float_dtype(df[col]):
            original_dtype = df[col].dtype
            try:
                df[col] = pd.to_numeric(df[col], errors='raise').astype(np.float32)
                if original_dtype != df[col].dtype:
                    print(f"  - Col {col} have already been converted from {original_dtype} to float32")
            except (ValueError, TypeError) as e:
                print(f"  - Warnings: Col {col} can't be converted from (Origin Type: {original_dtype})")
    return df