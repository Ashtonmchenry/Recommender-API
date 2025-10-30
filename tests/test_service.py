import json
from fastapi.testclient import TestClient
from service.app import app  # adjust if your app module differs

def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.text.lower().strip('"') in ("ok", "healthy")

def test_recommend_route_smoke(monkeypatch):
    c = TestClient(app)
    # minimal stub to avoid loading big models in unit test
    def fake_recommend(user_id: int, k: int = 5): return [1,2,3,4,5]
    monkeypatch.setenv("MODEL_PATH", "models/reco.joblib")
    # if your app calls a function, monkeypatch it; adjust import path accordingly

    from notebooks.M3_P1_2 import recommend_items_for_uid as recommend_for_user
    # example: monkeypatch.setattr("service.recommend.recommend_for_user", fake_recommend)
    r = c.get("/recommend", params={"user_id": 42, "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "items" in data and len(data["items"]) == 3
