import os
import sys

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from src.service.app import app  # noqa: E402

client = TestClient(app)


def test_batch_prediction_endpoint():
    payload = {
        "items": [{"features": [0.0] * 54 + [1.0, 1.0, 1.0]}] * 8,
        "n_jobs": 2,
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 8
    assert data["workers_used"] == 2
    assert data["status"] == "ok"


def test_batch_validation_error():
    payload = {
        "items": [{"features": [0.0] * 56}],
        "n_jobs": 1,
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 422


def test_pool_failure_resilience():
    bad_payload = {"items": [], "n_jobs": 4}
    resp = client.post("/predict/batch", json=bad_payload)
    assert resp.status_code == 422

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"


def test_drift_metrics_compliance():
    csv_path = "reports/LAB7/drift_metrics.csv"
    assert os.path.exists(csv_path), "Missing drift_metrics.csv"

    df = pd.read_csv(csv_path)
    clean_df = df[df["scenario"] == "clean_stream"]
    drift_df = df[df["scenario"] == "drift_stream"]

    assert clean_df["alert"].sum() == 0, "False alarms in clean stream"

    first_alert = drift_df[drift_df["alert"]].iloc[0]
    w_idx = first_alert["window_idx"]
    assert w_idx <= 3, f"Late alert on window {w_idx}"
