# Feature engineering placeholders for offline training/inference

# features.py
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleansing: types, NA/dup removal, rating bounds.
    Expects columns user_id, item_id, rating, timestamp (rating may be float).
    """
    out = df.copy()
    out = out.dropna(subset=["user_id", "item_id", "timestamp"])
    out = out.drop_duplicates(subset=["user_id", "item_id", "timestamp"], keep="last")
    out["user_id"] = out["user_id"].astype(int)
    out["item_id"] = out["item_id"].astype(int)
    if "rating" in out:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(0.0).clip(0, 5)
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype(int)
    return out

def user_item_activity(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (#events per user, #events per item)."""
    return df.groupby("user_id").size(), df.groupby("item_id").size()

def popularity(df: pd.DataFrame) -> pd.Series:
    """
    Simple popularity score per item (count or mean rating if available).
    Returns a Series indexed by item_id.
    """
    if "rating" in df:
        return df.groupby("item_id")["rating"].mean().sort_values(ascending=False)
    return df.groupby("item_id").size().sort_values(ascending=False)

def recency_feature(df: pd.DataFrame) -> pd.Series:
    """Scaled recency per interaction (0..1) by min-max on timestamp."""
    t = df["timestamp"].astype(float)
    if t.max() == t.min():
        return pd.Series(0.0, index=df.index, name="recency")
    return ((t - t.min()) / (t.max() - t.min())).rename("recency")

def chronological_cutoff(df: pd.DataFrame, q: float = 0.8) -> int:
    """Return the timestamp cutoff at quantile q (used for chronological splits)."""
    return int(np.quantile(df["timestamp"].values, q))

def eligible_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only users/items that appear in training in both sets (no cold-start in test).
    """
    train_users = set(train_df["user_id"])
    train_items = set(train_df["item_id"])
    test_df = test_df[test_df["user_id"].isin(train_users) & test_df["item_id"].isin(train_items)].copy()
    return train_df, test_df

def make_id_maps(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build contiguous 0..N-1 maps for users/items using sorted unique ids."""
    uid2idx = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    iid2idx = {m: i for i, m in enumerate(sorted(df["item_id"].unique()))}
    return uid2idx, iid2idx
