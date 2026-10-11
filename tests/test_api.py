from fastapi.testclient import TestClient

from src.engine.api import app


def test_health_and_analyze():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/analyze", json={"text": "Boeing faces FAA probe after quality lapses"})
    assert r.status_code == 200 and r.json()[0]["entity"] == "BA"
