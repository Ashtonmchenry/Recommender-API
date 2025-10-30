import pandas as pd
from recommender.features import basic_clean, popularity, recency_feature, chronological_cutoff, make_id_maps

def test_basic_clean_and_recency():
    df = pd.DataFrame({
        "user_id":[1,1,2], "item_id":[10,10,20], "rating":[5,4,5], "timestamp":[100,110,200]
    })
    out = basic_clean(df)
    assert out.dtypes["user_id"] == "int64"
    assert out.dtypes["item_id"] == "int64"
    assert out["rating"].between(0,5).all()
    r = recency_feature(out)
    assert (r.min() == 0.0) and (r.max() == 1.0)

def test_popularity_and_cutoff_maps():
    df = pd.DataFrame({
        "user_id":[1,1,2,3], "item_id":[10,11,10,11], "rating":[5,1,5,5], "timestamp":[10,20,30,40]
    })
    pop = popularity(df)
    assert set(pop.index) == {10,11}
    c = chronological_cutoff(df, 0.5)
    assert c in (20,30)  # middle quantile
    uid2idx, iid2idx = make_id_maps(df)
    assert set(uid2idx.values()) == set(range(len(uid2idx)))
    assert set(iid2idx.values()) == set(range(len(iid2idx)))
