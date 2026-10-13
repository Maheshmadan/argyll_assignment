"""Score failure risk for one device from raw sensor cycles.

`POST /predict` accepts the current cycle and at least three earlier cycles.
Lags are built with the same function as training. `POST /predict/stream` is
the same contract. `POST /predict/features` scores a matrix that already has
the training column names.

Each scored batch can be written to the `ai4i_deployment_monitoring` MLflow
experiment. Distribution drift is computed when the batch has at least 30
rows. If labels are included, precision, recall, F1, and ROC-AUC are logged
as well.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring import compare_to_reference, evaluate_predictions, flat_metrics
from src.paths import MODEL_PATH, tracking_uri
from src.preprocess import (
    SENSOR_COLS,
    TYPE_LEVELS,
    add_lag_features,
    design_matrix,
    drop_lag_warmup,
    machine_repetition_report,
)
from src.quality import (
    ASSUMPTIONS,
    ROW_KEY,
    find_issues,
    load_configuration,
    order_cycles,
    public_issues,
)

app = FastAPI(title="AI4I failure model", version="1.1.0")
_bundle: dict | None = None

_SENSOR_FIELDS = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}


class SensorRecord(BaseModel):
    """One cycle. Dataset column names and the short JSON names are both accepted."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    type: str = Field(validation_alias=AliasChoices("type", "Type"))
    air_temperature_k: float = Field(validation_alias=AliasChoices("air_temperature_k", "Air temperature [K]"))
    process_temperature_k: float = Field(validation_alias=AliasChoices("process_temperature_k", "Process temperature [K]"))
    rotational_speed_rpm: float = Field(validation_alias=AliasChoices("rotational_speed_rpm", "Rotational speed [rpm]"))
    torque_nm: float = Field(validation_alias=AliasChoices("torque_nm", "Torque [Nm]"))
    tool_wear_min: float = Field(validation_alias=AliasChoices("tool_wear_min", "Tool wear [min]"))
    timestamp: str | None = Field(default=None, validation_alias=AliasChoices("timestamp", "Timestamp"))
    product_id: str | None = Field(default=None, validation_alias=AliasChoices("product_id", "Product ID"))
    udi: int | None = Field(default=None, validation_alias=AliasChoices("udi", "UDI"))


class PredictRequest(BaseModel):
    device_id: str = Field(default="MILL_01", min_length=1)
    records: list[SensorRecord] = Field(min_length=1)
    labels: list[int] | None = None
    log_to_mlflow: bool = True


class FeaturePredictRequest(BaseModel):
    records: list[dict] = Field(min_length=1)
    labels: list[int] | None = None
    log_to_mlflow: bool = True


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Saved model not found at {MODEL_PATH}. Train it with python -m src.train.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def _validation_detail(exc: RequestValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"] if part != "body")
        parts.append(f"{location}: {error['msg']}")
    return "Invalid request. " + "; ".join(parts)


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc: RequestValidationError):
    del request
    return JSONResponse(status_code=400, content={"detail": _validation_detail(exc)})


def _frame_from_records(records: list[dict], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise HTTPException(status_code=400, detail=f"Record {index} must be an object of feature values.")
        missing = [name for name in feature_names if name not in record or record[name] is None]
        if missing:
            raise HTTPException(status_code=400, detail={"record": index, "missing_features": missing})
        try:
            rows.append({name: float(record[name]) for name in feature_names})
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Record {index} has a non-numeric feature.") from exc
    return pd.DataFrame(rows, columns=feature_names)


def _check_labels(labels: list[int] | None, n_rows: int) -> np.ndarray | None:
    if labels is None:
        return None
    if len(labels) != n_rows:
        raise HTTPException(status_code=400, detail=f"Expected {n_rows} labels, received {len(labels)}.")
    array = np.asarray(labels)
    if not np.isin(array, [0, 1]).all():
        raise HTTPException(status_code=400, detail="Labels must be 0 or 1.")
    return array.astype(int)


def _score(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    bundle = get_bundle()
    proba = bundle["estimator"].predict_proba(frame[bundle["feature_names"]])[:, 1]
    threshold = float(bundle["threshold"])
    pred = (proba >= threshold).astype(int)
    return pred, proba, threshold


def _monitor(frame: pd.DataFrame, pred: np.ndarray, proba: np.ndarray, labels: np.ndarray | None, enabled: bool) -> dict:
    bundle = get_bundle()
    drift = compare_to_reference(bundle["reference_distribution"], frame)
    performance = None
    if labels is not None:
        performance = evaluate_predictions(labels, proba, float(bundle["threshold"]))
    mlflow_run_id = None
    mlflow_error = None
    if enabled:
        try:
            mlflow.set_tracking_uri(tracking_uri())
            mlflow.set_experiment("ai4i_deployment_monitoring")
            with mlflow.start_run(run_name="prediction_batch") as run:
                mlflow.set_tags(
                    {
                        "model_name": bundle["model_name"],
                        "training_run_id": bundle.get("training_run_id", ""),
                        "stage": "deployment",
                    }
                )
                mlflow.log_metrics(
                    {
                        "batch_rows": float(len(frame)),
                        "predicted_positive_rate": float(np.mean(pred)),
                        "mean_probability": float(np.mean(proba)),
                    }
                )
                if drift.get("mean_psi") is not None:
                    mlflow.log_metric("mean_psi", float(drift["mean_psi"]))
                if performance is not None:
                    mlflow.log_metrics(flat_metrics("batch", performance))
                mlflow_run_id = run.info.run_id
        except Exception as exc:  # prediction must still succeed if tracking is down
            mlflow_error = str(exc)
    if drift.get("mean_psi_status") == "major_shift":
        drift["action"] = (
            "Retrain before treating these scores as current. "
            "The batch distribution has moved far from the training reference."
        )
    return {
        "distribution": drift,
        "performance": performance,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_error": mlflow_error,
    }


def _predictions(pred: np.ndarray, proba: np.ndarray, threshold: float, index: np.ndarray | None = None) -> list[dict]:
    rows = []
    for position, (label, probability) in enumerate(zip(pred, proba)):
        row = {
            "machine_failure": int(label),
            "failure_risk": float(probability),
            "probability": float(probability),
            "risk_band": risk_band(float(probability), threshold),
            "threshold": threshold,
        }
        if index is not None:
            row["source_row"] = int(index[position])
        rows.append(row)
    return rows


def _sensor_frame(device_id: str, records: list[SensorRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        payload = record.model_dump()
        row = {
            "Type": payload["type"],
            "timestamp": payload["timestamp"],
            "Product ID": payload["product_id"],
            "UDI": payload["udi"],
            "device_id": device_id,
        }
        for field, column in _SENSOR_FIELDS.items():
            row[column] = payload[field]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame[ROW_KEY] = np.arange(len(frame))
    return frame


def _iso(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.isoformat().replace("+00:00", "Z")


def _raw_predictions(pred, proba, threshold, ready: pd.DataFrame) -> list[dict]:
    rows = []
    for position, (label, probability) in enumerate(zip(pred, proba)):
        record = ready.iloc[position]
        row = {
            "machine_failure": int(label),
            "failure_risk": float(probability),
            "probability": float(probability),
            "risk_band": risk_band(float(probability), threshold),
            "threshold": threshold,
            "input_index": int(record["input_index"]),
        }
        timestamp = _iso(record["timestamp"]) if "timestamp" in ready.columns else None
        if timestamp is not None:
            row["timestamp"] = timestamp
        if "source_udi" in ready.columns and not pd.isna(record["source_udi"]):
            row["source_udi"] = int(record["source_udi"])
        rows.append(row)
    return rows


def _predict_raw(body: PredictRequest) -> dict:
    devices, events, configuration_issues = load_configuration()
    if configuration_issues:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Device configuration cannot be used.",
                "errors": public_issues(configuration_issues),
            },
        )
    frame = _sensor_frame(body.device_id, body.records)
    issues = find_issues(
        frame,
        devices,
        events,
        missing_policy="reject",
        apply_device_rules=True,
        default_device=body.device_id,
    )
    if issues:
        visible = public_issues(issues)
        message = visible[0]["message"] if len(visible) == 1 else f"{len(visible)} records failed validation."
        raise HTTPException(status_code=400, detail={"message": message, "errors": visible})

    labels = _check_labels(body.labels, len(frame))
    ordered, quality = order_cycles(frame)
    lagged = add_lag_features(ordered, machine_repetition_report(ordered))
    ready = drop_lag_warmup(lagged)
    if ready.empty:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Need the current cycle and at least 3 earlier cycles for this device. "
                    "The model uses lag 1, lag 2, and lag 3, so a single reading cannot be scored."
                ),
                "rows": int(len(ordered)),
                "minimum_rows": 4,
            },
        )
    source_index = ready["input_index"].to_numpy()
    features = design_matrix(ready)
    pred, proba, threshold = _score(features)
    aligned_labels = None if labels is None else labels[source_index]
    monitor = _monitor(features, pred, proba, aligned_labels, body.log_to_mlflow)
    predictions = _raw_predictions(pred, proba, threshold, ready)
    return {
        "device_id": body.device_id,
        "model_name": get_bundle()["model_name"],
        "failure_risk": predictions[-1],
        "predictions": predictions,
        "warmup_rows": int(len(ordered) - len(ready)),
        "data_quality": quality,
        "monitoring": monitor,
    }


@app.get("/health")
def health():
    bundle = get_bundle()
    devices, _, issues = load_configuration()
    return {
        "status": "ok" if not issues else "configuration_error",
        "model_name": bundle["model_name"],
        "threshold": bundle["threshold"],
        "n_features": len(bundle["feature_names"]),
        "test_metrics": bundle.get("test_metrics"),
        "model_path": str(MODEL_PATH),
        "devices": sorted(devices),
        "configuration_issues": public_issues(issues),
    }


@app.get("/features")
def features():
    bundle = get_bundle()
    return {
        "model_name": bundle["model_name"],
        "feature_names": bundle["feature_names"],
        "threshold": bundle["threshold"],
        "raw_sensor_fields": ["type", "timestamp", "product_id", "udi", * _SENSOR_FIELDS],
        "dataset_columns": ["Type", "timestamp", "Product ID", "UDI", *SENSOR_COLS],
        "type_levels": TYPE_LEVELS,
        "minimum_cycles": 4,
        "assumptions": ASSUMPTIONS,
    }


@app.post("/model/reload")
def reload_model():
    """Load a replacement model.joblib without restarting the process."""
    global _bundle
    _bundle = None
    bundle = get_bundle()
    return {
        "status": "reloaded",
        "model_name": bundle["model_name"],
        "threshold": bundle["threshold"],
        "training_run_id": bundle.get("training_run_id"),
    }


@app.post("/predict")
def predict(body: PredictRequest):
    """Score raw sensor cycles for one configured device."""
    return _predict_raw(body)


@app.post("/predict/stream")
def predict_stream(body: PredictRequest):
    """Same contract as POST /predict."""
    return _predict_raw(body)


@app.post("/predict/features")
def predict_features(body: FeaturePredictRequest):
    """Score rows that already use the training feature names."""
    bundle = get_bundle()
    scored = _frame_from_records(body.records, bundle["feature_names"])
    labels = _check_labels(body.labels, len(scored))
    pred, proba, threshold = _score(scored)
    monitor = _monitor(scored, pred, proba, labels, body.log_to_mlflow)
    return {
        "model_name": bundle["model_name"],
        "predictions": _predictions(pred, proba, threshold),
        "monitoring": monitor,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
