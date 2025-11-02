from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import service.app as app_module


def test_health_endpoints(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "1.2.3")

    assert app_module.health() == {"status": "ok", "version": "1.2.3"}
    assert app_module.healthz().body.decode() == "ok"


def test_recommend_json_route(monkeypatch):
    def fake_recommender(user_id: int, k: int, model_path=None):
        return [user_id + i for i in range(k)]

    monkeypatch.setattr(app_module, "_recommend_for_user", fake_recommender)

    payload = app_module.recommend(user_id=7, k=3)
    assert payload == {"user_id": 7, "items": [7, 8, 9]}


def test_recommend_plaintext_route(monkeypatch):
    def fake_recommender(user_id: int, k: int, model_path=None):
        return [1, 2, 3][:k]

    monkeypatch.setattr(app_module, "_recommend_for_user", fake_recommender)

    response = app_module.recommend_plain(user_id=99, k=2)
    assert response.body.decode() == "1,2"
