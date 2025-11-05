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

# ---------------- CLI/Reporting glue ----------------
import argparse
from pathlib import Path

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # unify common variants to snake_case
    rename_map = {}
    # users
    if "user_id" not in out.columns:
        if "userId" in out.columns: rename_map["userId"] = "user_id"
        elif "userid" in out.columns: rename_map["userid"] = "user_id"
    # items
    if "movie_id" not in out.columns:
        if "movieId" in out.columns: rename_map["movieId"] = "movie_id"
        elif "item_id" in out.columns: rename_map["item_id"] = "movie_id"
        elif "movieid" in out.columns: rename_map["movieid"] = "movie_id"
    # rating
    if "rating" not in out.columns:
        if "Rating" in out.columns: rename_map["Rating"] = "rating"
    # timestamps
    if "timestamp" not in out.columns:
        if "ts" in out.columns: rename_map["ts"] = "timestamp"
        elif "Timestamp" in out.columns: rename_map["Timestamp"] = "timestamp"

    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def _freq(series: pd.Series, topn: int = 20) -> pd.Series:
    s = series.value_counts(normalize=True).head(topn)
    s.index = s.index.astype(str)
    return s

def _drift_table(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    columns=("user_id", "movie_id", "rating"),
    thresholds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    metrics, flagged = detect_drift(ref_df, cur_df, columns=columns, thresholds=thresholds)
    rows = []
    for col in columns:
        if col in metrics:
            rows.append({
                "column": col,
                "psi": round(metrics[col]["psi"], 6),
                "kl": round(metrics[col]["kl"], 6),
                "psi_thresh": (thresholds or DEFAULT_THRESHOLDS)["psi"],
                "kl_thresh": (thresholds or DEFAULT_THRESHOLDS)["kl"],
                "flagged": (metrics[col]["psi"] > (thresholds or DEFAULT_THRESHOLDS)["psi"]) or
                           (metrics[col]["kl"]  > (thresholds or DEFAULT_THRESHOLDS)["kl"]),
            })
        else:
            rows.append({"column": col, "psi": None, "kl": None,
                         "psi_thresh": (thresholds or DEFAULT_THRESHOLDS)["psi"],
                         "kl_thresh": (thresholds or DEFAULT_THRESHOLDS)["kl"],
                         "flagged": None})
    tbl = pd.DataFrame(rows)
    tbl.attrs["overall_flagged"] = any(bool(r["flagged"]) for r in rows if r["flagged"] is not None)
    return tbl

def _plot_quick(ref_df: pd.DataFrame, cur_df: pd.DataFrame, out_dir: Path, topn: int = 20) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Items (movie_id)
    if "movie_id" in ref_df.columns and "movie_id" in cur_df.columns:
        b_items = _freq(ref_df["movie_id"], topn)
        c_items = _freq(cur_df["movie_id"], topn)
        ids = sorted(set(b_items.index) | set(c_items.index), key=lambda x: int(x))
        bx = b_items.reindex(ids, fill_value=0.0).values
        cx = c_items.reindex(ids, fill_value=0.0).values
        x = range(len(ids))

        plt.figure(figsize=(11, 5))
        plt.bar([i - 0.2 for i in x], bx, width=0.4, label="baseline")
        plt.bar([i + 0.2 for i in x], cx, width=0.4, label="current")
        plt.title(f"Top-{topn} item popularity (normalized)")
        plt.xlabel("movie_id")
        plt.ylabel("proportion")
        plt.xticks(list(x), ids, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "items_topN.png", dpi=150)
        plt.close()

    # --- Users
    if "user_id" in ref_df.columns and "user_id" in cur_df.columns:
        b_users = _freq(ref_df["user_id"], topn)
        c_users = _freq(cur_df["user_id"], topn)
        uids = sorted(set(b_users.index) | set(c_users.index), key=lambda x: int(x))
        bu = b_users.reindex(uids, fill_value=0.0).values
        cu = c_users.reindex(uids, fill_value=0.0).values
        x = range(len(uids))

        plt.figure(figsize=(11, 5))
        plt.bar([i - 0.2 for i in x], bu, width=0.4, label="baseline")
        plt.bar([i + 0.2 for i in x], cu, width=0.4, label="current")
        plt.title(f"Top-{topn} user activity (normalized)")
        plt.xlabel("user_id")
        plt.ylabel("proportion")
        plt.xticks(list(x), uids, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "users_topN.png", dpi=150)
        plt.close()

    # --- Ratings (optional)
    if "rating" in ref_df.columns and "rating" in cur_df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(ref_df["rating"].dropna().astype(float), bins=20, alpha=0.6, label="baseline", density=True)
        plt.hist(cur_df["rating"].dropna().astype(float), bins=20, alpha=0.6, label="current", density=True)
        plt.title("Rating distribution (density)")
        plt.xlabel("rating")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "ratings_hist.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Data drift report (PSI/KL + quick plots)")
    ap.add_argument("--baseline", required=True, help="CSV (e.g., data/ratings.csv)")
    ap.add_argument("--current",  required=True, help="CSV (e.g., out_eval/features.csv)")
    ap.add_argument("--columns",  default="user_id,item_id,rating",
                    help="comma-separated list; will map item_id→movie_id internally")
    ap.add_argument("--out",      default="out_eval/drift_summary.csv", help="where to write the table CSV")
    ap.add_argument("--plots",    default="out_eval/drift_plots",       help="directory for PNGs")
    ap.add_argument("--psi-threshold", type=float, default=DEFAULT_THRESHOLDS["psi"])
    ap.add_argument("--kl-threshold",  type=float, default=DEFAULT_THRESHOLDS["kl"])
    ap.add_argument("--topn",     type=int, default=20, help="top-N for popularity plots")
    args = ap.parse_args()

    ref_df = pd.read_csv(args.baseline)
    cur_df = pd.read_csv(args.current)

    # normalize common column names to this module's expected keys
    ref_df = _normalize_cols(ref_df)
    cur_df = _normalize_cols(cur_df)

    # map user’s "item_id" to our internal "movie_id" key for metrics
    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    cols = ["movie_id" if c == "item_id" else c for c in cols]

    thresholds = {"psi": args.psi_threshold, "kl": args.kl_threshold}
    table = _drift_table(ref_df, cur_df, columns=cols, thresholds=thresholds)

    # print table nicely
    try:
        from tabulate import tabulate
        print(tabulate(table, headers="keys", tablefmt="github", floatfmt=".6f"))
    except Exception:
        print(table.to_string(index=False))

    # write CSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    # plots
    _plot_quick(ref_df, cur_df, Path(args.plots), topn=args.topn)

if __name__ == "__main__":
    main()
