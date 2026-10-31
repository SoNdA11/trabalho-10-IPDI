import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import app


def test_status():
    client = app.test_client()
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.get_json()
    assert "model_ready" in data


def test_metrics():
    client = app.test_client()
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.get_json()
    assert "test_accuracy" in data


def test_classes():
    client = app.test_client()
    r = client.get("/api/classes")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data["classes"]) == 10