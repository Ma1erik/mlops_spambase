import json
import pytest
from fastapi.testclient import TestClient
from src.service.app import app
from src.service.registry import ModelRegistry

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_version" in data


def test_predict_valid_features():
    payload = {"features": [0.0] * 57}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert "model_version" in data


def test_contract_missing_field_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_contract_invalid_length_422():
    payload = {"features": [0.0] * 56}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_contract_negative_value_422():
    feats = [0.0] * 57
    feats[0] = -5.0
    response = client.post("/predict", json={"features": feats})
    assert response.status_code == 422


def test_contract_wrong_data_type_422():
    feats = [0.0] * 56 + ["not_a_number"]
    response = client.post("/predict", json={"features": feats})
    assert response.status_code == 422


def test_negative_control_hash_mismatch(tmp_path):
    fake_model_file = tmp_path / "fake_model.joblib"
    fake_model_file.write_bytes(b"corrupted_content")

    fake_registry = {
        "active_version": "v_fake",
        "models": {
            "v_fake": {
                "version": "v_fake",
                "model_type": "ml_joblib",
                "path": str(fake_model_file),
                "sha256": "0" * 64,
                "accuracy": 0.5,
                "f1_score": 0.5,
                "status": "рабочая",
            }
        },
    }
    reg_file = tmp_path / "registry.json"
    reg_file.write_text(json.dumps(fake_registry))

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ModelRegistry(registry_file=str(reg_file))
