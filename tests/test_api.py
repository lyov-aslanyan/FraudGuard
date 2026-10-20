import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app import app, feature_names

client = TestClient(app, raise_server_exceptions=False)

def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "FraudGuard API is running"


def test_predict_valid_transaction():
    response = client.post(
        "/predict",
        json={"features": [0.0] * len(feature_names)},
    )

    assert response.status_code == 200

    data = response.json()

    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("fraud", "legitimate")


def test_predict_wrong_feature_count():
    response = client.post(
        "/predict",
        json={"features": [0.0, 1.0]},
    )

    assert response.status_code == 422


def test_predict_missing_features():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422

def test_rejects_nan_value():
    features = ["0.0"] * 30
    features[0] = '"NaN"'

    response = client.post(
        "/predict",
        content=(
            '{"features": ['
            + ",".join(features)
            + "]}"
        ),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_rejects_infinity_value():
    features = ["0.0"] * 30
    features[0] = '"Infinity"'

    response = client.post(
        "/predict",
        content=(
            '{"features": ['
            + ",".join(features)
            + "]}"
        ),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422