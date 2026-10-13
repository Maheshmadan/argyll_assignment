"""Load AI4I cycles, reject bad rows, fill missing values, and build lag features.

Duplicates, impossible sensor values, broken timestamps, and device-configuration
conflicts are removed before imputation. The published file has none of these.
The rules still run so a later extract is held to the same checks.


Missing numeric values are filled only after a holdout check. Mean, median,
empirical resampling, and predictive mean matching are scored on a masked
copy of each column. The selected method is the one whose filled column stays
closest to the original distribution (Kolmogorov-Smirnov, mean, and standard
deviation). The training frame is left unchanged when the source file has no
nulls; the check still records which method would have been used.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_LEVELS = ["H", "L", "M"]
TARGET_COL = "Machine failure"
# These flags are components of the failure label, so they are not inputs.
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLS = ["UDI", "Product ID"]
LAGS = (1, 2, 3)
MACHINE_KEY = "MILL_01"
DISTRIBUTION_KS_LIMIT = 0.05
DISTRIBUTION_MOMENT_LIMIT = 0.05
METHOD_PREFERENCE = {"pmm": 0, "empirical": 1, "median": 2, "mean": 3}


def safe_feature_name(name: str) -> str:
    """XGBoost rejects feature names that contain brackets."""
    cleaned = name
    for old, new in (
        (" [K]", "_k"),
        (" [rpm]", "_rpm"),
        (" [Nm]", "_nm"),
        (" [min]", "_min"),
    ):
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.replace(" ", "_").replace("[", "").replace("]", "").replace("<", "")
    return cleaned.lower()


def lag_feature_name(column: str, lag: int) -> str:
    return f"{column}__lag{lag}"


def lag_feature_names(columns: list[str] | tuple[str, ...] = SENSOR_COLS, lags: tuple[int, ...] = LAGS) -> list[str]:
    return [lag_feature_name(column, lag) for column in columns for lag in lags]


def load_raw(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ID_COLS + ["Type"] + SENSOR_COLS + [TARGET_COL] + LEAKAGE_COLS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return frame


def null_report(frame: pd.DataFrame) -> dict:
    counts = frame.isna().sum()
    return {
        "rows": int(len(frame)),
        "columns": {column: int(counts[column]) for column in frame.columns},
        "total_nulls": int(counts.sum()),
    }


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _design_matrix_for_imputation(frame: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=frame.index)
    for column in SENSOR_COLS:
        design[column] = pd.to_numeric(frame[column], errors="coerce")
    for level in TYPE_LEVELS:
        design[f"Type_{level}"] = (frame["Type"] == level).astype(float)
    return design


def _fill_predictors(design: pd.DataFrame) -> pd.DataFrame:
    filled = design.copy()
    for column in filled.columns:
        if filled[column].isna().any():
            median = filled[column].median()
            if pd.isna(median):
                median = 0.0
            filled[column] = filled[column].fillna(median)
    return filled


def impute_mean(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> pd.DataFrame:
    del rng
    work = frame.copy()
    observed = work[column].dropna()
    if observed.empty or work[column].isna().sum() == 0:
        return work
    work[column] = work[column].fillna(float(observed.mean()))
    return work


def impute_median(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> pd.DataFrame:
    del rng
    work = frame.copy()
    observed = work[column].dropna()
    if observed.empty or work[column].isna().sum() == 0:
        return work
    work[column] = work[column].fillna(float(observed.median()))
    return work


def impute_empirical(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> pd.DataFrame:
    work = frame.copy()
    missing = work[column].isna()
    donors = work.loc[~missing, column].to_numpy(dtype=float)
    if donors.size == 0 or int(missing.sum()) == 0:
        return work
    work.loc[missing, column] = rng.choice(donors, size=int(missing.sum()), replace=True)
    return work


def impute_pmm(frame: pd.DataFrame, column: str, rng: np.random.Generator, k: int = 10) -> pd.DataFrame:
    """Fill `column` with real observed donors nearest in prediction space.

    Donors are actual observed values, so the filled marginal stays on the
    support of the original feature instead of collapsing to a single mean.
    """
    work = frame.copy()
    missing = work[column].isna()
    n_missing = int(missing.sum())
    observed = ~missing
    donors = work.loc[observed, column].to_numpy(dtype=float)
    if n_missing == 0 or donors.size == 0:
        return work
    if donors.size < 20:
        work.loc[missing, column] = rng.choice(donors, size=n_missing, replace=True)
        return work

    design = _fill_predictors(_design_matrix_for_imputation(work))
    predictors = [name for name in design.columns if name != column]
    model = Ridge(alpha=1.0)
    model.fit(design.loc[observed, predictors], work.loc[observed, column].astype(float))
    pred_obs = model.predict(design.loc[observed, predictors])
    pred_mis = model.predict(design.loc[missing, predictors])
    k_use = min(k, donors.size)
    distances = np.abs(pred_mis[:, None] - pred_obs[None, :])
    neighbors = np.argpartition(distances, k_use - 1, axis=1)[:, :k_use]
    draws = np.array(
        [donors[int(rng.choice(neighbors[row]))] for row in range(n_missing)],
        dtype=float,
    )
    work.loc[missing, column] = draws
    return work


IMPUTERS: dict[str, Callable] = {
    "mean": impute_mean,
    "median": impute_median,
    "empirical": impute_empirical,
    "pmm": impute_pmm,
}


def _distribution_gap(original: np.ndarray, filled: np.ndarray) -> dict:
    original = np.asarray(original, dtype=float)
    filled = np.asarray(filled, dtype=float)
    scale = float(np.std(original)) + 1e-9
    ks_stat = float(ks_2samp(original, filled).statistic)
    mean_shift = abs(float(np.mean(filled)) - float(np.mean(original))) / scale
    std_shift = abs(float(np.std(filled)) - float(np.std(original))) / scale
    return {
        "ks": ks_stat,
        "mean_shift": float(mean_shift),
        "std_shift": float(std_shift),
        "preserved": bool(
            ks_stat <= DISTRIBUTION_KS_LIMIT
            and mean_shift <= DISTRIBUTION_MOMENT_LIMIT
            and std_shift <= DISTRIBUTION_MOMENT_LIMIT
        ),
    }


def score_imputation_methods(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    mask_fraction: float = 0.2,
    repeats: int = 3,
    random_state: int = 42,
) -> dict:
    """Mask observed values and measure how each imputer changes the column."""
    columns = list(columns or SENSOR_COLS)
    rng = np.random.default_rng(random_state)
    per_column = {}
    for column in columns:
        source = pd.to_numeric(frame[column], errors="coerce")
        observed_index = source.dropna().index.to_numpy()
        if observed_index.size < 50:
            per_column[column] = {"error": "not enough observed values to score imputers"}
            continue
        totals = {name: [] for name in IMPUTERS}
        mask_size = max(1, int(round(observed_index.size * mask_fraction)))
        for _ in range(repeats):
            chosen = rng.choice(observed_index, size=mask_size, replace=False)
            original = source.to_numpy(dtype=float)
            finite = np.isfinite(original)
            masked = frame.copy()
            masked[column] = source
            masked.loc[chosen, column] = np.nan
            for name, imputer in IMPUTERS.items():
                filled = imputer(masked, column, rng)[column].to_numpy(dtype=float)
                totals[name].append(_distribution_gap(original[finite], filled[finite]))
        method_scores = {}
        for name, gaps in totals.items():
            method_scores[name] = {
                "ks": float(np.mean([gap["ks"] for gap in gaps])),
                "mean_shift": float(np.mean([gap["mean_shift"] for gap in gaps])),
                "std_shift": float(np.mean([gap["std_shift"] for gap in gaps])),
            }
            method_scores[name]["preserved"] = bool(
                method_scores[name]["ks"] <= DISTRIBUTION_KS_LIMIT
                and method_scores[name]["mean_shift"] <= DISTRIBUTION_MOMENT_LIMIT
                and method_scores[name]["std_shift"] <= DISTRIBUTION_MOMENT_LIMIT
            )
        eligible = [name for name, score in method_scores.items() if score["preserved"]]
        pool = eligible or list(method_scores)
        selected = min(pool, key=lambda name: (method_scores[name]["ks"], METHOD_PREFERENCE[name]))
        per_column[column] = {
            "mask_fraction": mask_fraction,
            "repeats": repeats,
            "methods": method_scores,
            "selected_method": selected,
            "distribution_preserved": bool(method_scores[selected]["preserved"]),
        }
    return per_column


def apply_imputation(
    frame: pd.DataFrame,
    method_by_column: dict[str, str],
    random_state: int = 42,
) -> pd.DataFrame:
    work = frame.copy()
    rng = np.random.default_rng(random_state)
    ordered = sorted(
        [column for column in method_by_column if column in work.columns],
        key=lambda column: int(work[column].isna().sum()),
    )
    for column in ordered:
        if work[column].isna().sum() == 0:
            continue
        method = method_by_column[column]
        if method not in IMPUTERS:
            raise ValueError(f"Unknown imputation method {method!r} for {column}")
        work = IMPUTERS[method](work, column, rng)
        if work[column].isna().any():
            raise RuntimeError(f"Imputation left nulls in {column}")
    return work


def audit_and_impute(frame: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """Return a completed frame and a JSON-ready audit.

    When a column has nulls, the method chosen on the observed-value holdout
    is applied, then the completed column is compared with the values that
    were already present. When a column is complete, the frame is not modified.
    """
    report = {"nulls_before": null_report(frame), "columns": {}, "rows_changed": False}
    present_sensors = [column for column in SENSOR_COLS if column in frame.columns]
    simulation = score_imputation_methods(frame, present_sensors, random_state=random_state)
    method_by_column = {
        column: details["selected_method"]
        for column, details in simulation.items()
        if "selected_method" in details
    }
    completed = frame
    if int(frame[present_sensors].isna().sum().sum()) > 0:
        before = frame.copy()
        completed = apply_imputation(frame, method_by_column, random_state=random_state)
        report["rows_changed"] = True
        for column in present_sensors:
            missing = before[column].isna()
            if int(missing.sum()) == 0:
                continue
            observed = before.loc[~missing, column].to_numpy(dtype=float)
            filled = completed[column].to_numpy(dtype=float)
            report["columns"][column] = {
                "n_missing": int(missing.sum()),
                "method": method_by_column.get(column),
                "completed_vs_observed": _distribution_gap(observed, filled),
            }
    else:
        for column in present_sensors:
            report["columns"][column] = {
                "n_missing": 0,
                "method": None,
                "note": "No nulls, so the column was not modified.",
            }
    report["method_selection"] = simulation
    report["nulls_after"] = null_report(completed)
    return completed, _to_builtin(report)


def _machine_id_columns(frame: pd.DataFrame) -> list[str]:
    """Identifier columns only. `Machine failure` is the label, not a machine id."""
    blocked = set(ID_COLS + SENSOR_COLS + LEAKAGE_COLS + [TARGET_COL, "Type", "machine_key"])
    names = {"machine_id", "machine", "device_id", "asset_id", "equipment_id"}
    found = []
    for column in frame.columns:
        if column in blocked:
            continue
        lowered = column.lower().replace(" ", "_")
        if lowered in names or lowered.endswith("_machine_id"):
            found.append(column)
    return found


def machine_repetition_report(frame: pd.DataFrame) -> dict:
    """Decide whether lag features can be built from a repeated machine stream."""
    product_unique = int(frame["Product ID"].nunique()) if "Product ID" in frame.columns else None
    type_counts = frame["Type"].value_counts().to_dict() if "Type" in frame.columns else {}
    machine_columns = _machine_id_columns(frame)
    udi_monotonic = bool(frame["UDI"].is_monotonic_increasing) if "UDI" in frame.columns else False
    repeated_machine_column = next(
        (column for column in machine_columns if 1 < frame[column].nunique() < len(frame)),
        None,
    )
    if repeated_machine_column:
        strategy = "group_by_machine_column"
        machine_key = repeated_machine_column
    else:
        # Product ID is unique per cycle. Type is a quality variant (L/M/H),
        # and UDI increases on every row, so the file is one milling stream.
        strategy = "single_process_stream"
        machine_key = MACHINE_KEY
    return _to_builtin(
        {
            "rows": int(len(frame)),
            "unique_product_id": product_unique,
            "type_counts": {str(key): int(value) for key, value in type_counts.items()},
            "machine_columns_found": machine_columns,
            "udi_monotonic": udi_monotonic,
            "strategy": strategy,
            "machine_key": machine_key,
            "lags": list(LAGS),
            "lag_source_columns": list(SENSOR_COLS),
        }
    )


def add_lag_features(frame: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Add prior-cycle sensor values for the repeated machine stream.

    Rows are ordered by UDI inside each machine. The first cycles of a machine
    do not have a full lag window and are removed by the caller.
    """
    if report is None:
        report = machine_repetition_report(frame)
    work = frame.copy()
    if "UDI" not in work.columns:
        work["UDI"] = np.arange(1, len(work) + 1)
    if report["strategy"] == "group_by_machine_column":
        key_column = report["machine_key"]
        work = work.sort_values([key_column, "UDI"]).reset_index(drop=True)
        group_key = work[key_column]
    else:
        work = work.sort_values("UDI").reset_index(drop=True)
        group_key = pd.Series(MACHINE_KEY, index=work.index, name="machine_key")
    work["machine_key"] = group_key.to_numpy()
    grouped = work.groupby("machine_key", sort=False)
    for column in SENSOR_COLS:
        for lag in LAGS:
            work[lag_feature_name(column, lag)] = grouped[column].shift(lag)
    return work


def drop_lag_warmup(frame: pd.DataFrame) -> pd.DataFrame:
    lag_columns = [column for column in lag_feature_names() if column in frame.columns]
    return frame.dropna(subset=lag_columns).reset_index(drop=True)


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    data = {f"Type_{level}": (frame["Type"] == level).astype(float) for level in TYPE_LEVELS}
    for column in list(SENSOR_COLS) + lag_feature_names():
        data[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    matrix = pd.DataFrame(data, index=frame.index)
    matrix.columns = [safe_feature_name(column) for column in matrix.columns]
    if matrix.columns.duplicated().any():
        raise ValueError(f"Feature names are not unique: {matrix.columns.tolist()}")
    if matrix.isna().any().any():
        missing = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(f"Feature matrix still has nulls in {missing}")
    return matrix


def prepare_training_frame(path, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series, dict]:
    # Local import: src.quality imports the column constants defined above.
    from src.quality import clean_training_frame

    raw = load_raw(path)
    cleaned, quality_report = clean_training_frame(raw)
    imputed, imputation_report = audit_and_impute(cleaned, random_state=random_state)
    repetition = machine_repetition_report(imputed)
    lagged = drop_lag_warmup(add_lag_features(imputed, repetition))
    features = design_matrix(lagged)
    target = lagged[TARGET_COL].astype(int)
    report = {
        "quality": quality_report,
        "imputation": imputation_report,
        "repetition": repetition,
        "rows_after_lag_warmup": int(len(features)),
        "feature_names": features.columns.tolist(),
        "excluded_columns": ID_COLS + LEAKAGE_COLS + [TARGET_COL, "machine_key"],
    }
    return features, target, _to_builtin(report)
