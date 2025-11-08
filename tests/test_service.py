from pathlib import Path
from types import SimpleNamespace
import sys

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import service.app as app_module


class DummyRegistry:
    def __init__(self, version: str | None):
        self._version = version

    def active_version(self) -> str | None:
        return self._version


def test_health_endpoints(monkeypatch):
    monkeypatch.setattr(app_module, "_REGISTRY", DummyRegistry("1.2.3"))

    assert app_module.health() == {"status": "ok", "version": "1.2.3"}
    assert app_module.healthz() == "ok"


def test_recommend_json_route(monkeypatch):
    def fake_recommender(user_id: int, k: int, model_version=None):
        return [user_id + i for i in range(k)], "v-test"

    monkeypatch.setattr(app_module, "_recommend_for_user", fake_recommender)

    payload = app_module.recommend(user_id=7, k=3)
    assert payload == {
        "user_id": 7,
        "items": [7, 8, 9],
        "model_version": "v-test",
        "generated_at": payload["generated_at"],
    }


def test_recommend_plaintext_route(monkeypatch):
    def fake_recommender(user_id: int, k: int, model_version=None):
        return [1, 2, 3][:k], "v-plain"

    monkeypatch.setattr(app_module, "_recommend_for_user", fake_recommender)

    response = app_module.recommend_plain(user_id=99, k=2)
    assert response == "1,2"


def _build_registry(tmp_path: Path, version: str = "v-test") -> Path:
    registry_root = tmp_path / "registry"
    version_dir = registry_root / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model = SimpleNamespace(
        user_map={1: 0},
        item_map={101: 0, 202: 1},
        user_factors=np.array([[0.3, 0.9]]),
        item_factors=np.array([[1.0, 0.0], [0.1, 1.2]]),
    )
    joblib.dump(model, version_dir / app_module.MODEL_ARTIFACT_NAME)
    (version_dir / app_module.MODEL_METADATA_NAME).write_text(
        "created_at: 2024-01-01T00:00:00+00:00\nmetrics:\n  hr@20: 0.42\n",
        encoding="utf-8",
    )
    return registry_root


def test_switch_and_recommendation_flow(tmp_path):
    registry_root = _build_registry(tmp_path)

    try:
        app_module.configure_registry(registry_root)

        # Without an active version we should fall back to defaults.
        items, version = app_module._recommend_for_user(user_id=99, k=3)
        assert version is None
        assert items == app_module.DEFAULT_RECOMMENDATIONS[:3]

        # Activate and ensure the new model is used.
        response = app_module.switch("v-test")
        assert response["status"] == "ok"

        recs, active_version = app_module._recommend_for_user(user_id=1, k=2)
        assert active_version == "v-test"
        assert recs[0] in {101, 202}
    finally:
        # Restore the default registry to avoid cross-test contamination.
        app_module.configure_registry(app_module.MODEL_REGISTRY_ROOT, version=app_module.INITIAL_MODEL_VERSION)


def test_switch_missing_version(tmp_path):
    app_module.configure_registry(tmp_path)
    try:
        with pytest.raises(app_module.HTTPException) as excinfo:
            app_module.switch("does-not-exist")
        assert excinfo.value.status_code == 404
    finally:
        app_module.configure_registry(app_module.MODEL_REGISTRY_ROOT, version=app_module.INITIAL_MODEL_VERSION)
