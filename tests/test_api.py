"""API contract for raw sensor scoring and the engineered-feature endpoint."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from deployment.app import app
from src.paths import DATA_PATH, MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train the model with python -m src.train")


def _records(count: int = 4, *, reverse_time: bool = False, dataset_names: bool = False) -> list[dict]:
    frame = pd.read_csv(DATA_PATH).head(count)
    records = []
    for offset, row in enumerate(frame.to_dict(orient="records")):
        minute = count - 1 - offset if reverse_time else offset
        timestamp = (pd.Timestamp("2020-01-02T00:00:00Z") + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if dataset_names:
            records.append(
                {
                    "timestamp": timestamp,
                    "Type": row["Type"],
                    "Product ID": row["Product ID"],
                    "UDI": int(row["UDI"]),
                    "Air temperature [K]": row["Air temperature [K]"],
                    "Process temperature [K]": row["Process temperature [K]"],
                    "Rotational speed [rpm]": row["Rotational speed [rpm]"],
                    "Torque [Nm]": row["Torque [Nm]"],
                    "Tool wear [min]": row["Tool wear [min]"],
                }
            )
        else:
            records.append(
                {
                    "timestamp": timestamp,
                    "type": row["Type"],
                    "product_id": row["Product ID"],
                    "udi": int(row["UDI"]),
                    "air_temperature_k": row["Air temperature [K]"],
                    "process_temperature_k": row["Process temperature [K]"],
                    "rotational_speed_rpm": row["Rotational speed [rpm]"],
                    "torque_nm": row["Torque [Nm]"],
                    "tool_wear_min": row["Tool wear [min]"],
                }
            )
    return records


def _messages(response) -> str:
    detail = response.json()["detail"]
    if isinstance(detail, str):
        return detail
    errors = detail.get("errors", [])
    return detail.get("message", "") + " " + " ".join(item["message"] for item in errors)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_predict_scores_raw_sensors(client):
    response = client.post("/predict", json={"device_id": "MILL_01", "records": _records(), "log_to_mlflow": False})
    assert response.status_code == 200
    body = response.json()
    risk = body["failure_risk"]
    assert body["warmup_rows"] == 3
    assert len(body["predictions"]) == 1
    assert risk == body["predictions"][-1]
    assert 0.0 <= risk["failure_risk"] <= 1.0
    assert risk["probability"] == risk["failure_risk"]
    assert risk["machine_failure"] in {0, 1}
    assert risk["risk_band"] in {"low", "medium", "high"}
    assert risk["input_index"] == 3
    assert body["data_quality"]["reordered"] is False
    assert body["model_name"] in {"xgboost", "knn"}


def test_dataset_column_names_and_stream_alias(client):
    response = client.post(
        "/predict/stream",
        json={"records": _records(dataset_names=True), "log_to_mlflow": False},
    )
    assert response.status_code == 200
    assert response.json()["device_id"] == "MILL_01"
    assert response.json()["failure_risk"]["failure_risk"] >= 0


def test_out_of_order_timestamps_are_sorted(client):
    response = client.post("/predict", json={"records": _records(reverse_time=True), "log_to_mlflow": False})
    assert response.status_code == 200
    body = response.json()
    assert body["data_quality"]["reordered"] is True
    assert body["data_quality"]["order_by"] == "timestamp"
    assert body["failure_risk"]["input_index"] == 0


def test_single_cycle_explains_the_lag_window(client):
    response = client.post("/predict", json={"records": _records(1), "log_to_mlflow": False})
    assert response.status_code == 400
    assert "lag 1, lag 2, and lag 3" in _messages(response)


def test_invalid_sensor_duplicate_and_config_errors(client):
    base = _records(1)[0]
    cases = [
        ({**base, "torque_nm": -1}, "outside the accepted range"),
        ({**base, "type": "Q"}, "Expected one of"),
        ({**base, "product_id": "M999", "type": "L"}, "does not start with Type"),
        ({**base, "timestamp": "yesterday"}, "ISO-8601"),
        ({**base, "timestamp": "2019-01-01T00:00:00Z"}, "was installed"),
        ({**base, "device_id_override": "NOPE"}, "is not in the device configuration"),
    ]
    for record, phrase in cases:
        device_id = "MILL_01"
        payload = dict(record)
        if "device_id_override" in payload:
            device_id = payload.pop("device_id_override")
        response = client.post("/predict", json={"device_id": device_id, "records": [payload], "log_to_mlflow": False})
        assert response.status_code == 400
        assert phrase in _messages(response)

    duplicate = _records(2)
    duplicate[1]["product_id"] = duplicate[0]["product_id"]
    response = client.post("/predict", json={"records": duplicate, "log_to_mlflow": False})
    assert response.status_code == 400
    assert "share Product ID" in _messages(response)

    limited = dict(base)
    limited.update({"type": "L", "product_id": "L1", "rotational_speed_rpm": 5000, "timestamp": "2020-07-01T00:00:00Z"})
    response = client.post("/predict", json={"device_id": "MILL_02", "records": [limited], "log_to_mlflow": False})
    assert response.status_code == 400
    assert "configuration limit" in _messages(response)

    retired = dict(limited)
    retired["rotational_speed_rpm"] = 1500
    retired["timestamp"] = "2021-06-01T00:00:00Z"
    response = client.post("/predict", json={"device_id": "MILL_02", "records": [retired], "log_to_mlflow": False})
    assert response.status_code == 400
    assert "retire event" in _messages(response)

    wrong_type = dict(limited)
    wrong_type.update({"type": "M", "product_id": "M1", "rotational_speed_rpm": 1500})
    response = client.post("/predict", json={"device_id": "MILL_02", "records": [wrong_type], "log_to_mlflow": False})
    assert response.status_code == 400
    assert "not configured to run" in _messages(response)


def test_schema_errors_are_readable(client):
    response = client.post("/predict", json={"records": [{"type": "L"}], "log_to_mlflow": False})
    assert response.status_code == 400
    assert "Invalid request" in response.json()["detail"]
    assert "torque_nm" in response.json()["detail"]

    engineered = json.loads((MODEL_PATH.parent / "sample_request.json").read_text(encoding="utf-8"))
    engineered["log_to_mlflow"] = False
    response = client.post("/predict", json=engineered)
    assert response.status_code == 400
    assert "type" in response.json()["detail"]


def test_feature_endpoint_scores_the_saved_matrix(client):
    payload = json.loads((MODEL_PATH.parent / "sample_request.json").read_text(encoding="utf-8"))
    payload["log_to_mlflow"] = False
    response = client.post("/predict/features", json=payload)
    assert response.status_code == 200
    prediction = response.json()["predictions"][0]
    assert 0.0 <= prediction["failure_risk"] <= 1.0
    assert prediction["risk_band"] in {"low", "medium", "high"}


def test_sample_payload_file(client):
    payload = json.loads((MODEL_PATH.parent / "sample_sensor_request.json").read_text(encoding="utf-8"))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["failure_risk"]["input_index"] == 3
    assert response.json()["data_quality"]["reordered"] is False


def test_health_features_and_reload(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "MILL_01" in health.json()["devices"]
    assert "MILL_02" in health.json()["devices"]

    described = client.get("/features")
    assert described.status_code == 200
    assert described.json()["minimum_cycles"] == 4
    assert described.json()["assumptions"]

    reloaded = client.post("/model/reload")
    assert reloaded.status_code == 200
    assert reloaded.json()["status"] == "reloaded"
    assert reloaded.json()["model_name"] == health.json()["model_name"]
