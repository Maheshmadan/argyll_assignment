"""Distribution drift and classification metrics for training and deployment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(pred)),
        "support_positive": int(np.sum(y_true)),
        "support_negative": int(np.sum(y_true == 0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, proba))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    grid = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, (proba >= threshold).astype(int), zero_division=0) for threshold in grid]
    index = int(np.argmax(scores))
    return float(grid[index]), float(scores[index])


def reference_distribution(frame: pd.DataFrame, n_bins: int = 10, sample_size: int = 2000, seed: int = 42) -> dict:
    """Equal-width training histograms used later for population stability."""
    rng = np.random.default_rng(seed)
    reference = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        lo = float(np.min(values))
        hi = float(np.max(values))
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
        take = min(sample_size, values.size)
        sample = values[rng.choice(values.size, size=take, replace=False)]
        reference[column] = {
            "edges": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sample": [float(value) for value in sample],
        }
    return reference


def _psi(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    expected = np.asarray(expected_counts, dtype=float)
    actual = np.asarray(actual_counts, dtype=float)
    expected_share = expected / max(expected.sum(), 1e-12)
    actual_share = actual / max(actual.sum(), 1e-12)
    expected_share = np.clip(expected_share, 1e-6, None)
    actual_share = np.clip(actual_share, 1e-6, None)
    expected_share = expected_share / expected_share.sum()
    actual_share = actual_share / actual_share.sum()
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def psi_status(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "major_shift"


def compare_to_reference(reference: dict, frame: pd.DataFrame, min_rows: int = 30) -> dict:
    if len(frame) < min_rows:
        return {
            "status": "skipped",
            "reason": f"Need at least {min_rows} rows to compare distributions.",
            "rows": int(len(frame)),
        }
    features = {}
    for column, spec in reference.items():
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=float)
        edges = np.asarray(spec["edges"], dtype=float)
        open_edges = edges.copy()
        open_edges[0] = -np.inf
        open_edges[-1] = np.inf
        counts, _ = np.histogram(values, bins=open_edges)
        psi = _psi(np.asarray(spec["counts"]), counts)
        ks_stat = float(ks_2samp(np.asarray(spec["sample"], dtype=float), values).statistic)
        features[column] = {
            "psi": psi,
            "psi_status": psi_status(psi),
            "ks": ks_stat,
            "reference_mean": spec["mean"],
            "batch_mean": float(np.mean(values)),
            "reference_std": spec["std"],
            "batch_std": float(np.std(values)),
        }
    psi_values = [item["psi"] for item in features.values()]
    mean_psi = float(np.mean(psi_values)) if psi_values else None
    return {
        "status": "ok",
        "rows": int(len(frame)),
        "mean_psi": mean_psi,
        "mean_psi_status": psi_status(mean_psi) if mean_psi is not None else None,
        "features": features,
    }


def flat_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None and not isinstance(value, bool):
            flat[f"{prefix}_{key}"] = float(value)
    return flat
