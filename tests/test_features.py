from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import pytest

from recommender.features import (
    basic_clean,
    chronological_cutoff,
    eligible_overlap,
    make_id_maps,
    popularity,
    recency_feature,
    user_item_activity,
)

# NEW: import the split you actually rely on in training/eval
from recommender.train import last_item_holdout, reindex


def test_basic_clean_normalizes_types_and_bounds():
    dirty = pd.DataFrame(
        {
            "user_id": ["1", "1", "2", None],
            "item_id": ["10", "10", "20", "30"],
            "rating": ["5", "bad", 9, 4],
            "timestamp": [100, 110, 200, None],
        }
    )

    cleaned = basic_clean(dirty)

    assert cleaned["user_id"].dtype.kind in ("i", "u")
    assert cleaned["item_id"].dtype.kind in ("i", "u")
    assert cleaned["rating"].between(0, 5).all()
    assert cleaned["timestamp"].dtype.kind in ("i", "u")


    recency = recency_feature(cleaned)
    assert recency.min() == 0.0
    assert recency.max() == 1.0


# NEW: allow 'ts' and ensure it is normalized to 'timestamp'
def test_basic_clean_accepts_ts_and_normalizes_to_timestamp():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 11, 20],
            "rating":  [5,  3,  4],
            "ts":      [10, 20, 30],  # note: 'ts' instead of 'timestamp'
        }
    )
    out = basic_clean(df)
    assert "timestamp" in out.columns
    assert out["timestamp"].dtype.kind in ("i", "u")  # coerced to int
    # keep alignment
    assert out.sort_values(["user_id","item_id"])["timestamp"].tolist() == [10,20,30]


def test_activity_cutoff_and_id_maps():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "item_id": [10, 11, 10, 11],
            "rating": [5, 1, 5, 5],
            "timestamp": [10, 20, 30, 40],
        }
    )

    users, items = user_item_activity(df)
    assert users.to_dict() == {1: 2, 2: 1, 3: 1}
    assert items.to_dict() == {10: 2, 11: 2}

    cut = chronological_cutoff(df, 0.5)
    assert 10 <= cut <= 40

    uid2idx, iid2idx = make_id_maps(df)
    assert sorted(uid2idx.values()) == list(range(len(uid2idx)))
    assert sorted(iid2idx.values()) == list(range(len(iid2idx)))


def test_eligible_overlap_and_popularity_filtering():
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 11, 10],
            "rating": [5, 3, 4],
            "timestamp": [1, 2, 3],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [1, 3],
            "item_id": [10, 12],
            "rating": [5, 2],
            "timestamp": [4, 5],
        }
    )

    _, filtered = eligible_overlap(train, test)
    assert filtered["user_id"].tolist() == [1]
    assert filtered["item_id"].tolist() == [10]

    pop = popularity(train)
    assert list(pop.index) == [10, 11]


# NEW: end-to-end sanity for chronological last-item holdout (the split your training uses)
def test_last_item_holdout_latest_item_per_user():
    raw = pd.DataFrame(
        {
            "user_id": [1,1,1, 2,2, 3],
            "item_id": [10,11,12,20,21,30],
            "rating":  [4, 5, 3, 5, 4, 5],
            "ts":      [10,50,40,30,60, 5],   # latest: u1->11(ts50), u2->21(ts60), u3->30(ts5)
        }
    )
    clean = basic_clean(raw)         # normalizes ts→timestamp, types, dup removal
    indexed = reindex(clean)  # adds uidx/iidx
    train, test = last_item_holdout(indexed)

    # one test row per user
    assert len(test) == indexed["uidx"].nunique()

    # verify chosen item is truly the most recent per user
    latest = indexed.sort_values(["uidx","timestamp"]).groupby("uidx").tail(1)
    assert set(zip(test["uidx"], test["true_item"])) == set(zip(latest["uidx"], latest["iidx"]))

    # train should not contain those last (user, item) pairs
    merged = latest.merge(train, how="inner", on=["uidx","iidx"])
    assert merged.empty
