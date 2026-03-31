import yaml
from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["db_connected"] is True  # AsyncMock.execute() succeeds silently


def test_version(client: TestClient) -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == config["model_version"]
    assert "api_version" in data


def test_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content
