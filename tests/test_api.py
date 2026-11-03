"""API: só roda os testes de classificação se os modelos já foram treinados."""
import pytest

import config as C
from app import app

client = app.test_client()
trained = pytest.mark.skipif(not (C.MODELS / "rf_eurosat_ms.joblib").exists(), reason="modelos não treinados")


def test_status():
    r = client.get("/api/status")
    assert r.status_code == 200 and "models" in r.get_json()


def test_index_served():
    assert client.get("/").status_code == 200


@trained
def test_metrics_are_test_set_metrics():
    d = client.get("/api/metrics").get_json()
    assert 0 < d["eurosat"]["accuracy"] <= 1
    assert sum(map(sum, d["eurosat"]["confusion_matrix"])) == d["eurosat"]["n_test"]


@trained
def test_classify_test_sample_returns_real_prediction():
    r = client.get("/api/eurosat/classify/0").get_json()
    assert r["class_name"] in C.EUROSAT_CLASSES
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert r["inference_ms"] > 0 and len(r["features"]) > 0


@trained
def test_upload_rejects_unknown_format():
    import io
    r = client.post("/api/eurosat/upload", data={"file": (io.BytesIO(b"x"), "a.gif")})
    assert r.status_code == 400


@trained
def test_scene_window_and_pixel():
    r = client.post("/api/scenes/cbers/window", json={"x": 500, "y": 500, "size": 64}).get_json()
    assert r["size"] == 64 and abs(sum(r["fractions"].values()) - 1) < 1e-6
    p = client.get("/api/scenes/cbers/pixel?x=520&y=520").get_json()
    assert "valid" in p
