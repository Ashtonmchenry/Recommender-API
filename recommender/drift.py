"""Data drift utilities and self-checks for ALS recommender."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _bin_edges(series: pd.Series, bins: int = 10) -> np.ndarray:
    """Return histogram bin edges derived from the series quantiles."""

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    values = np.unique(np.quantile(series.dropna(), quantiles))
    if values.size < 2:
        val = float(values[0]) if values.size == 1 else 0.0
        return np.array([val - 0.5, val + 0.5])
    return values


def population_stability_index(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Compute PSI between two 1-D numeric distributions."""

    edges = _bin_edges(ref, bins)
    ref_hist, _ = np.histogram(ref.dropna(), bins=edges)
    cur_hist, _ = np.histogram(cur.dropna(), bins=edges)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def kl_divergence(ref: pd.Series, cur: pd.Series, bins: int = 20) -> float:
    """Compute KL divergence between two numeric distributions."""

    edges = _bin_edges(ref, bins)
    p_hist, _ = np.histogram(ref.dropna(), bins=edges)
    q_hist, _ = np.histogram(cur.dropna(), bins=edges)

    p_pct = p_hist / max(p_hist.sum(), 1)
    q_pct = q_hist / max(q_hist.sum(), 1)

    p_pct = np.where(p_pct == 0, 1e-6, p_pct)
    q_pct = np.where(q_pct == 0, 1e-6, q_pct)
    return float(np.sum(p_pct * np.log(p_pct / q_pct)))


DEFAULT_THRESHOLDS: Dict[str, float] = {"psi": 0.2, "kl": 0.5}


def detect_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    columns: Iterable[str] = ("user_id", "movie_id", "rating"),
    thresholds: Dict[str, float] | None = None,
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Return (metrics, drift_flag) comparing reference vs. current data."""

    metrics: Dict[str, Dict[str, float]] = {}
    limits = thresholds or DEFAULT_THRESHOLDS

    drift = False
    for column in columns:
        if column not in ref_df or column not in cur_df:
            continue

        ref_series = ref_df[column].astype(float)
        cur_series = cur_df[column].astype(float)

        psi = population_stability_index(ref_series, cur_series)
        kl = kl_divergence(ref_series, cur_series)
        metrics[column] = {"psi": psi, "kl": kl}

        if psi > limits["psi"] or kl > limits["kl"]:
            drift = True

    return metrics, drift


def _run_self_checks() -> None:
    """Execute lightweight drift checks for local validation."""

    ref = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "movie_id": [10, 11, 10, 11],
        "rating": [3, 4, 3, 4],
    })
    cur = pd.DataFrame({
        "user_id": [1, 2, 1, 2],
        "movie_id": [10, 11, 10, 11],
        "rating": [3, 4, 3, 4],
    })
    metrics, flagged = detect_drift(ref, cur)
    assert set(metrics) == {"user_id", "movie_id", "rating"}
    assert flagged is False

    ref_shift = pd.DataFrame({
        "user_id": [1] * 50 + [2] * 50,
        "movie_id": [10] * 100,
        "rating": [4.0] * 100,
    })
    cur_shift = pd.DataFrame({
        "user_id": [9] * 100,
        "movie_id": [99] * 100,
        "rating": [1.0] * 100,
    })
    shift_metrics, shift_flagged = detect_drift(ref_shift, cur_shift)
    assert shift_flagged is True
    for values in shift_metrics.values():
        assert values["psi"] >= DEFAULT_THRESHOLDS["psi"] or values["kl"] >= DEFAULT_THRESHOLDS["kl"]


if __name__ == "__main__":
    _run_self_checks()
    print("Drift self-checks passed.")
