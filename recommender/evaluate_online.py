# Join reco_responses with watch events to compute an online KPI proxy

import pandas as pd

def proxy_kpi(responses_csv: str, watches_csv: str, horizon_min: int = 10) -> float:
    r = pd.read_csv(responses_csv)   # columns: request_id, user_id, movie_ids (array or JSON), ts
    w = pd.read_csv(watches_csv)     # columns: user_id, movie_id, ts

    # explode movie_ids to rows if stored as arrays; if JSON string, json.loads then explode
    if r["movie_ids"].dtype == object:
        r = r.explode("movie_ids")

    # join on user_id within time horizon
    merged = r.merge(w, on="user_id", suffixes=("_rec", "_watch"))
    merged = merged[ (merged["ts_watch"] - merged["ts_rec"]).between(0, horizon_min*60) ]
    merged["hit"] = (merged["movie_ids"] == merged["movie_id"]).astype(int)

    # proxy: % of requests with at least one hit
    hits = merged.groupby("request_id")["hit"].max()
    return float(hits.mean())
