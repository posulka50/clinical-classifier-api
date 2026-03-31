from fastapi.testclient import TestClient

VALID_FEATURES = [0.0] * 30


def test_valid_prediction(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/api/v1/predict",
        json={"features": VALID_FEATURES},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("benign", "malignant")
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["probabilities"].keys()) == {"benign", "malignant"}
    assert "model_version" in data


def test_missing_api_key(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json={"features": VALID_FEATURES})
    assert response.status_code == 403


def test_invalid_api_key(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={"features": VALID_FEATURES},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_missing_features_field(client: TestClient, auth_headers: dict) -> None:
    response = client.post("/api/v1/predict", json={}, headers=auth_headers)
    assert response.status_code == 422


def test_wrong_number_of_features_too_few(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/api/v1/predict", json={"features": [0.0] * 29}, headers=auth_headers
    )
    assert response.status_code == 422


def test_wrong_number_of_features_too_many(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/api/v1/predict", json={"features": [0.0] * 31}, headers=auth_headers
    )
    assert response.status_code == 422


def test_feature_out_of_range(client: TestClient, auth_headers: dict) -> None:
    features = [0.0] * 30
    features[0] = 15.0
    response = client.post(
        "/api/v1/predict", json={"features": features}, headers=auth_headers
    )
    assert response.status_code == 422


def test_probabilities_sum_to_one(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/api/v1/predict", json={"features": VALID_FEATURES}, headers=auth_headers
    )
    assert response.status_code == 200
    probs = response.json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3
