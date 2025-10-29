"""
clean, normalize data
"""

import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # minimal transformation: drop NA, enforce types
    df = df.dropna(subset=[c for c in df.columns if c in ("user_id","movie_id")])
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype(float)
    return df
