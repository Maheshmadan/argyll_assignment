"""Train XGBoost and KNN, keep the stronger test model, and log MLflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring import (
    best_f1_threshold,
    compare_to_reference,
    evaluate_predictions,
    flat_metrics,
    reference_distribution,
)
from src.paths import ARTIFACT_DIR, DATA_PATH, MODEL_PATH, tracking_uri
from src.preprocess import TARGET_COL, prepare_training_frame

RANDOM_STATE = 42
TRAIN_FRACTION = 0.8
VAL_FRACTION_OF_TRAIN = 0.85


def chronological_split(features: pd.DataFrame, target: pd.Series, train_fraction: float = TRAIN_FRACTION):
    cut = int(len(features) * train_fraction)
    if cut <= 0 or cut >= len(features):
        raise ValueError("Train fraction does not leave both a train and a test block.")
    return (
        features.iloc[:cut].reset_index(drop=True),
        features.iloc[cut:].reset_index(drop=True),
        target.iloc[:cut].reset_index(drop=True),
        target.iloc[cut:].reset_index(drop=True),
    )


def build_xgboost(y: pd.Series) -> tuple[XGBClassifier, float]:
    positives = max(int((y == 1).sum()), 1)
    negatives = int((y == 0).sum())
    scale_pos_weight = negatives / positives
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model, float(scale_pos_weight)


def build_knn() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=31, weights="distance", metric="minkowski", n_jobs=-1),
            ),
        ]
    )


def _fit_and_threshold(name: str, x_fit, y_fit, x_val, y_val, x_train, y_train):
    if name == "xgboost":
        model, scale_pos_weight = build_xgboost(y_fit)
        model.fit(x_fit, y_fit)
        val_proba = model.predict_proba(x_val)[:, 1]
        threshold, val_f1 = best_f1_threshold(y_val, val_proba)
        refit, scale_pos_weight = build_xgboost(y_train)
        refit.fit(x_train, y_train)
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos_weight,
            "threshold": threshold,
        }
        return refit, threshold, val_f1, params
    model = build_knn()
    model.fit(x_fit, y_fit)
    val_proba = model.predict_proba(x_val)[:, 1]
    threshold, val_f1 = best_f1_threshold(y_val, val_proba)
    refit = build_knn()
    refit.fit(x_train, y_train)
    params = {"n_neighbors": 31, "weights": "distance", "threshold": threshold}
    return refit, threshold, val_f1, params


def train(data_path=DATA_PATH) -> dict:
    features, target, prep_report = prepare_training_frame(data_path, random_state=RANDOM_STATE)
    x_train, x_test, y_train, y_test = chronological_split(features, target)
    val_cut = int(len(x_train) * VAL_FRACTION_OF_TRAIN)
    x_fit, x_val = x_train.iloc[:val_cut], x_train.iloc[val_cut:]
    y_fit, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]
    for split_name, labels in (
        ("train", y_train),
        ("validation", y_val),
        ("threshold_fit", y_fit),
        ("test", y_test),
    ):
        counts = labels.value_counts().to_dict()
        if 0 not in counts or 1 not in counts:
            raise RuntimeError(f"{split_name} split is missing a class: {counts}")

    reference = reference_distribution(x_train)
    drift = compare_to_reference(reference, x_test, min_rows=30)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("ai4i_training")

    candidates = []
    with mlflow.start_run(run_name="initial_xgboost_knn") as parent:
        mlflow.log_params(
            {
                "split": "chronological_80_20",
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "train_failure_rate": float(y_train.mean()),
                "test_failure_rate": float(y_test.mean()),
                "n_features": int(x_train.shape[1]),
                "machine_strategy": prep_report["repetition"]["strategy"],
                "lags": "1,2,3",
                "target": TARGET_COL,
            }
        )
        mlflow.set_tag("selection_metric", "test_f1")
        (ARTIFACT_DIR / "preprocess_report.json").write_text(json.dumps(prep_report, indent=2), encoding="utf-8")
        (ARTIFACT_DIR / "train_test_drift.json").write_text(json.dumps(drift, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(ARTIFACT_DIR / "preprocess_report.json"))
        mlflow.log_artifact(str(ARTIFACT_DIR / "train_test_drift.json"))
        if drift.get("mean_psi") is not None:
            mlflow.log_metric("test_mean_psi", float(drift["mean_psi"]))

        for name in ("xgboost", "knn"):
            with mlflow.start_run(run_name=name, nested=True):
                model, threshold, val_f1, params = _fit_and_threshold(
                    name, x_fit, y_fit, x_val, y_val, x_train, y_train
                )
                test_proba = model.predict_proba(x_test)[:, 1]
                train_proba = model.predict_proba(x_train)[:, 1]
                test_metrics = evaluate_predictions(y_test, test_proba, threshold)
                train_metrics = evaluate_predictions(y_train, train_proba, threshold)
                mlflow.log_params({key: value for key, value in params.items()})
                mlflow.log_metric("val_f1", val_f1)
                mlflow.log_metrics(flat_metrics("test", test_metrics))
                mlflow.log_metrics(flat_metrics("train", train_metrics))
                candidates.append(
                    {
                        "name": name,
                        "model": model,
                        "threshold": threshold,
                        "val_f1": val_f1,
                        "params": params,
                        "test_metrics": test_metrics,
                        "train_metrics": train_metrics,
                    }
                )

        def sort_key(item):
            roc = item["test_metrics"]["roc_auc"]
            return (item["test_metrics"]["f1"], -1.0 if roc is None else roc)

        best = max(candidates, key=sort_key)
        comparison = {
            item["name"]: {
                "threshold": item["threshold"],
                "val_f1": item["val_f1"],
                "train": item["train_metrics"],
                "test": item["test_metrics"],
                "params": item["params"],
            }
            for item in candidates
        }
        bundle = {
            "model_name": best["name"],
            "estimator": best["model"],
            "threshold": best["threshold"],
            "feature_names": x_train.columns.tolist(),
            "target": TARGET_COL,
            "reference_distribution": reference,
            "imputation_methods": {
                column: details.get("selected_method")
                for column, details in prep_report["imputation"]["method_selection"].items()
            },
            "test_metrics": best["test_metrics"],
            "comparison": comparison,
            "machine_strategy": prep_report["repetition"]["strategy"],
            "training_run_id": parent.info.run_id,
            "split": {
                "type": "chronological",
                "train_fraction": TRAIN_FRACTION,
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
            },
        }
        joblib.dump(bundle, MODEL_PATH)
        comparison_path = ARTIFACT_DIR / "model_comparison.json"
        public_comparison = {
            "best_model": best["name"],
            "selection_metric": "test_f1_then_roc_auc",
            "models": comparison,
            "train_test_drift_mean_psi": drift.get("mean_psi"),
            "train_test_drift_status": drift.get("mean_psi_status"),
        }
        comparison_path.write_text(json.dumps(public_comparison, indent=2), encoding="utf-8")
        sample = x_test.iloc[0].to_dict()
        sample_payload = {"records": [{key: float(value) for key, value in sample.items()}]}
        (ARTIFACT_DIR / "sample_request.json").write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_metric("best_test_f1", best["test_metrics"]["f1"])
        if best["test_metrics"]["roc_auc"] is not None:
            mlflow.log_metric("best_test_roc_auc", best["test_metrics"]["roc_auc"])
        mlflow.set_tag("best_model", best["name"])

    summary = {
        "best_model": best["name"],
        "threshold": best["threshold"],
        "test_metrics": best["test_metrics"],
        "comparison": {
            name: {"test_f1": payload["test"]["f1"], "test_roc_auc": payload["test"]["roc_auc"], "test_recall": payload["test"]["recall"], "test_precision": payload["test"]["precision"]}
            for name, payload in comparison.items()
        },
        "model_path": str(MODEL_PATH),
        "tracking_uri": tracking_uri(),
        "run_id": bundle["training_run_id"],
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_failure_rate": float(y_train.mean()),
        "test_failure_rate": float(y_test.mean()),
        "nulls_before": prep_report["imputation"]["nulls_before"]["total_nulls"],
        "imputation_applied": prep_report["imputation"]["rows_changed"],
        "machine_strategy": prep_report["repetition"]["strategy"],
        "n_features": int(x_train.shape[1]),
        "mean_test_psi": drift.get("mean_psi"),
    }
    (ARTIFACT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
