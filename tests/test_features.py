from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from recommender.features import (
    basic_clean,
    chronological_cutoff,
    eligible_overlap,
    make_id_maps,
    popularity,
    recency_feature,
    user_item_activity,
)


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

    assert cleaned["user_id"].dtype == "int64"
    assert cleaned["item_id"].dtype == "int64"
    assert cleaned["rating"].between(0, 5).all()
    assert cleaned["timestamp"].dtype == "int64"

    recency = recency_feature(cleaned)
    assert recency.min() == 0.0
    assert recency.max() == 1.0


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
