# Simple drift checks for distributions

from __future__ import annotations
import numpy as np
import pandas as pd

def _bin_edges(x: pd.Series, bins: int = 10):
    qs = np.linspace(0, 1, bins+1)
    return np.unique(np.quantile(x.dropna(), qs))

def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for one numeric series."""
    edges = _bin_edges(ref, bins)
    ref_pct, _ = np.histogram(ref.dropna(), bins=edges, density=True)
    cur_pct, _ = np.histogram(cur.dropna(), bins=edges, density=True)
    # avoid zeros
    ref_pct = np.where(ref_pct==0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct==0, 1e-6, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def kl_divergence(p: pd.Series, q: pd.Series, bins: int = 20) -> float:
    edges = _bin_edges(p, bins)
    p_hist, _ = np.histogram(p.dropna(), bins=edges, density=True)
    q_hist, _ = np.histogram(q.dropna(), bins=edges, density=True)
    p_hist = np.where(p_hist==0, 1e-6, p_hist)
    q_hist = np.where(q_hist==0, 1e-6, q_hist)
    return float(np.sum(p_hist * np.log(p_hist/q_hist)))

DEFAULT_THRESHOLDS = {
    "psi": 0.2,      # >0.2 suggests moderate shift; >0.4 strong
    "kl": 0.5
}

def detect_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, cols=("user_id","movie_id")) -> dict:
    report = {}
    for c in cols:
        if c not in ref_df or c not in cur_df: 
            continue
        try:
            p = ref_df[c].astype(float)
            q = cur_df[c].astype(float)
            report[c] = {
                "psi": psi(p,q),
                "kl": kl_divergence(p,q),
            }
        except Exception as e:
            report[c] = {"error": str(e)}
    return report

def any_drift(report: dict, thresholds: dict = None) -> bool:
    th = thresholds or DEFAULT_THRESHOLDS
    for c, m in report.items():
        if isinstance(m, dict) and (m.get("psi", 0) > th["psi"] or m.get("kl", 0) > th["kl"]):
            return True
    return False
