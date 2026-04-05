from __future__ import annotations

from fastapi.testclient import TestClient

from anima_turing_radar.api import create_app


def test_api_health_and_predict() -> None:
    app = create_app()
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    payload = {
        "pulses": [[float(i), 1000.0, 1.0, 0.0, -40.0] for i in range(16)],
        "algorithm": "kmeans",
    }
    p = client.post("/predict", json=payload)
    assert p.status_code == 200
    body = p.json()
    assert len(body["labels"]) == 16
