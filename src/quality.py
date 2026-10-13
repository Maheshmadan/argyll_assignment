"""Reject bad sensor cycles before imputation or scoring.

Historical files drop the bad rows and keep a count. A live request is rejected
so the caller can correct it. Missing sensors stay in the historical file for
imputation; a live request must include every sensor.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import DEVICE_CONFIG_PATH, MAINTENANCE_EVENTS_PATH
from src.preprocess import MACHINE_KEY, SENSOR_COLS, TYPE_LEVELS

# Physical limits. They contain the published AI4I ranges and reject garbage
# such as a negative torque or a temperature of 0 K.
SENSOR_BOUNDS = {
    "Air temperature [K]": (250.0, 350.0),
    "Process temperature [K]": (250.0, 400.0),
    "Rotational speed [rpm]": (1.0, 10000.0),
    "Torque [Nm]": (0.01, 500.0),
    "Tool wear [min]": (0.0, 10000.0),
}
DEVICE_LIMIT_FIELDS = {
    "Rotational speed [rpm]": "max_rotational_speed_rpm",
    "Torque [Nm]": "max_torque_nm",
    "Tool wear [min]": "max_tool_wear_min",
}
KNOWN_EVENT_TYPES = {"install", "tool_change", "inspection", "retire"}
ROW_KEY = "_quality_row"
ASSUMPTIONS = [
    "The UCI file has no clock time. UDI is the cycle order. A timestamp column, when present, overrides UDI.",
    "Naive timestamps are read as UTC.",
    "The UCI file has no machine id. Product ID is one cycle, and Type is the product quality variant L, M, or H. Batch training treats the file as device MILL_01 unless a device id column is present.",
    "TWF, HDF, PWF, OSF, and RNF are pieces of the failure label and are not model inputs.",
    "Live scoring needs the current cycle and three earlier cycles so lag 1, lag 2, and lag 3 match training.",
    "Missing sensors are imputed only while preparing a historical file. A live request must include every sensor.",
    "Duplicate UDI, Product ID, or device timestamp values are removed from historical files and rejected on the API.",
    "Maintenance events record install, tool change, inspection, and retire. Install time and retire time constrain readings. Tool changes and inspections are operational records, not model features.",
]


def normalize_types(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "Type" in work.columns:
        work["Type"] = work["Type"].map(lambda value: value if pd.isna(value) else str(value).strip().upper())
    return work


def _blank(value) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _row_number(row: dict) -> int:
    return int(row[ROW_KEY])


def _issue(row: int | None, field: str, message: str, drop_rows: list[int], kind: str) -> dict:
    return {
        "row": row,
        "field": field,
        "message": message,
        "drop_rows": drop_rows,
        "kind": kind,
    }


def public_issues(issues: list[dict]) -> list[dict]:
    return [{"row": issue["row"], "field": issue["field"], "message": issue["message"]} for issue in issues]


def _id_key(value):
    if _blank(value):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        return int(value)
    return str(value).strip()


def load_configuration(
    config_path: Path | None = None,
    maintenance_path: Path | None = None,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Load device limits and maintenance events.

    A problem in these files is returned to the caller. Devices that fail their
    own checks are left out of the map so a bad row cannot be treated as valid.
    """
    config_path = Path(config_path or DEVICE_CONFIG_PATH)
    maintenance_path = Path(maintenance_path or MAINTENANCE_EVENTS_PATH)
    issues: list[dict] = []
    devices: dict = {}
    empty_events = pd.DataFrame(columns=["device_id", "event_time", "event_type", "component", "notes"])

    if not config_path.exists():
        issues.append(_issue(None, "device_config", f"Device configuration file is missing: {config_path}", [], "configuration"))
        return devices, empty_events, issues
    config = pd.read_csv(config_path)
    required = [
        "device_id",
        "allowed_types",
        "status",
        "process",
        "installed_at",
        "max_rotational_speed_rpm",
        "max_torque_nm",
        "max_tool_wear_min",
    ]
    missing = [column for column in required if column not in config.columns]
    if missing:
        issues.append(_issue(None, "device_config", f"Device configuration is missing columns: {missing}", [], "configuration"))
        return devices, empty_events, issues

    for position, record in enumerate(config.to_dict(orient="records")):
        device_id = str(record["device_id"]).strip()
        allowed = [part.strip().upper() for part in str(record["allowed_types"]).split(",") if part.strip()]
        status = str(record["status"]).strip().lower()
        installed = pd.to_datetime(record["installed_at"], utc=True, errors="coerce", format="mixed")
        limits = {}
        limit_ok = True
        for field in DEVICE_LIMIT_FIELDS.values():
            try:
                limits[field] = float(record[field])
            except (TypeError, ValueError):
                limit_ok = False
        problems = []
        if not device_id:
            problems.append("device_id is blank")
        if device_id in devices:
            problems.append(f"device_id {device_id} is duplicated")
        if not allowed:
            problems.append("allowed_types is empty")
        unknown_types = [value for value in allowed if value not in TYPE_LEVELS]
        if unknown_types:
            problems.append(f"allowed_types has values outside {TYPE_LEVELS}: {unknown_types}")
        if status not in {"active", "retired"}:
            problems.append(f"status {status!r} must be active or retired")
        if pd.isna(installed):
            problems.append(f"installed_at {record['installed_at']!r} is not an ISO-8601 time")
        if not limit_ok:
            problems.append("sensor limits must be numbers")
        if problems:
            issues.append(
                _issue(position, "device_config", f"Device row {position} ({device_id or 'blank'}): " + "; ".join(problems), [], "configuration")
            )
            continue
        devices[device_id] = {
            "device_id": device_id,
            "allowed_types": allowed,
            "status": status,
            "process": str(record["process"]).strip(),
            "installed_at": installed,
            **limits,
        }

    if not maintenance_path.exists():
        issues.append(_issue(None, "maintenance", f"Maintenance event file is missing: {maintenance_path}", [], "configuration"))
        return devices, empty_events, issues
    events = pd.read_csv(maintenance_path)
    event_columns = ["device_id", "event_time", "event_type", "component", "notes"]
    missing_events = [column for column in event_columns if column not in events.columns]
    if missing_events:
        issues.append(_issue(None, "maintenance", f"Maintenance file is missing columns: {missing_events}", [], "configuration"))
        return devices, empty_events, issues
    events = events.copy()
    events["device_id"] = events["device_id"].map(lambda value: str(value).strip())
    events["event_type"] = events["event_type"].map(lambda value: str(value).strip())
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce", format="mixed")
    kept_events = []
    for position, record in enumerate(events.to_dict(orient="records")):
        problems = []
        if record["device_id"] not in devices:
            problems.append(f"device_id {record['device_id']!r} is not in the device configuration")
        if record["event_type"] not in KNOWN_EVENT_TYPES:
            problems.append(f"event_type {record['event_type']!r} must be one of {sorted(KNOWN_EVENT_TYPES)}")
        if pd.isna(record["event_time"]):
            problems.append("event_time is not an ISO-8601 time")
        if problems:
            issues.append(_issue(position, "maintenance", f"Maintenance row {position}: " + "; ".join(problems), [], "configuration"))
            continue
        kept_events.append(record)
    events = pd.DataFrame(kept_events, columns=event_columns) if kept_events else empty_events

    for device_id, spec in devices.items():
        device_events = events[events["device_id"] == device_id] if not events.empty else events
        installs = device_events[device_events["event_type"] == "install"] if not device_events.empty else device_events
        retires = device_events[device_events["event_type"] == "retire"] if not device_events.empty else device_events
        if len(installs) > 1:
            issues.append(_issue(None, "maintenance", f"Device {device_id} has more than one install event.", [], "configuration"))
        elif len(installs) == 1 and installs.iloc[0]["event_time"] != spec["installed_at"]:
            issues.append(
                _issue(
                    None,
                    "maintenance",
                    f"Device {device_id} install event {installs.iloc[0]['event_time'].isoformat()} does not match installed_at {spec['installed_at'].isoformat()}.",
                    [],
                    "configuration",
                )
            )
        if len(retires) > 1:
            issues.append(_issue(None, "maintenance", f"Device {device_id} has more than one retire event.", [], "configuration"))
    return devices, events, issues


def _parsed_timestamps(frame: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    if "timestamp" not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"), []
    raw = frame["timestamp"]
    missing = raw.map(_blank)
    if bool(missing.all()):
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"), []
    issues = []
    if bool(missing.any()):
        blank_rows = [int(frame.at[idx, ROW_KEY]) for idx in frame.index[missing]]
        issues.append(
            _issue(
                None,
                "timestamp",
                "Send a timestamp on every record, or omit timestamps and cycles will be scored in request order.",
                blank_rows,
                "timestamp",
            )
        )
    parsed = pd.to_datetime(raw.where(~missing), utc=True, errors="coerce", format="mixed")
    bad = ~missing & parsed.isna()
    for idx in frame.index[bad]:
        row_number = int(frame.at[idx, ROW_KEY])
        issues.append(
            _issue(
                row_number,
                "timestamp",
                f"Record {row_number} timestamp {raw.at[idx]!r} is not a valid ISO-8601 time.",
                [row_number],
                "timestamp",
            )
        )
    return parsed, issues


def _duplicate_issues(frame: pd.DataFrame, parsed: pd.Series, order_values: pd.Series) -> list[dict]:
    """Flag repeated keys. The earliest row is the one a historical file keeps."""
    issues = []
    row_numbers = frame[ROW_KEY].astype(int)
    checks: list[tuple[str, pd.Series]] = []
    if "UDI" in frame.columns:
        checks.append(("UDI", frame["UDI"].map(_id_key)))
    if "Product ID" in frame.columns:
        checks.append(("Product ID", frame["Product ID"].map(_id_key)))
    timestamp_keys = []
    for idx in frame.index:
        stamp = parsed.at[idx]
        if pd.isna(stamp):
            timestamp_keys.append(None)
        else:
            device = frame.at[idx, "device_id"] if "device_id" in frame.columns and not _blank(frame.at[idx, "device_id"]) else MACHINE_KEY
            timestamp_keys.append((str(device), stamp.isoformat()))
    checks.append(("timestamp", pd.Series(timestamp_keys, index=frame.index)))

    for field, keys in checks:
        groups: dict = {}
        for idx, key in keys.items():
            if key is None:
                continue
            groups.setdefault(key, []).append((order_values.at[idx], int(row_numbers.at[idx])))
        for key, members in groups.items():
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda item: (item[0], item[1]))
            rows = [row for _, row in members]
            shown = ", ".join(str(row) for row in rows)
            label = f"{key!r}" if field != "timestamp" else key[1]
            issues.append(
                _issue(
                    None,
                    field,
                    f"Records {shown} share {field} {label}.",
                    rows[1:],
                    "duplicate",
                )
            )
    return issues


def _device_for_row(row: dict, default_device: str) -> str:
    if "device_id" in row and not _blank(row.get("device_id")):
        return str(row["device_id"]).strip()
    return default_device


def _value_issues(
    frame: pd.DataFrame,
    devices: dict,
    events: pd.DataFrame,
    parsed: pd.Series,
    *,
    missing_policy: str,
    apply_device_rules: bool,
    default_device: str,
) -> list[dict]:
    issues = []
    records = frame.to_dict(orient="records")
    parsed_by_row = {int(frame.at[idx, ROW_KEY]): parsed.at[idx] for idx in frame.index}
    unknown_devices: dict[str, list[int]] = {}
    inactive_devices: dict[str, list[int]] = {}

    for row in records:
        row_number = _row_number(row)
        device_id = _device_for_row(row, default_device)
        spec = devices.get(device_id) if apply_device_rules else None
        if apply_device_rules and spec is None:
            unknown_devices.setdefault(device_id, []).append(row_number)
        elif apply_device_rules and spec["status"] != "active":
            inactive_devices.setdefault(device_id, []).append(row_number)
            spec = None

        type_value = row.get("Type")
        if _blank(type_value):
            issues.append(_issue(row_number, "Type", f"Record {row_number} is missing Type.", [row_number], "invalid"))
            type_value = None
        elif type_value not in TYPE_LEVELS:
            issues.append(
                _issue(
                    row_number,
                    "Type",
                    f"Record {row_number} Type is {type_value!r}. Expected one of {TYPE_LEVELS}.",
                    [row_number],
                    "invalid",
                )
            )
            type_value = None
        elif spec is not None and type_value not in spec["allowed_types"]:
            issues.append(
                _issue(
                    row_number,
                    "Type",
                    f"Record {row_number} Type is {type_value!r}, which device {device_id} is not configured to run. Allowed types: {', '.join(spec['allowed_types'])}.",
                    [row_number],
                    "invalid",
                )
            )

        product_id = row.get("Product ID")
        if type_value is not None and not _blank(product_id) and str(product_id).strip()[0] != type_value:
            issues.append(
                _issue(
                    row_number,
                    "Product ID",
                    f"Record {row_number} Product ID {str(product_id).strip()!r} does not start with Type {type_value!r}.",
                    [row_number],
                    "invalid",
                )
            )

        numbers = {}
        for column, (low, high) in SENSOR_BOUNDS.items():
            raw_value = row.get(column, np.nan)
            if _blank(raw_value):
                if missing_policy == "reject":
                    issues.append(
                        _issue(
                            row_number,
                            column,
                            f"Record {row_number} is missing {column}. Live scoring needs every sensor. Historical gaps are imputed only while preparing a training file.",
                            [row_number],
                            "invalid",
                        )
                    )
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                issues.append(_issue(row_number, column, f"Record {row_number} {column} is not a number.", [row_number], "invalid"))
                continue
            numbers[column] = value
            if value < low or value > high:
                issues.append(
                    _issue(
                        row_number,
                        column,
                        f"Record {row_number} {column} is {value}, outside the accepted range {low} to {high}.",
                        [row_number],
                        "invalid",
                    )
                )
                continue
            if spec is not None and column in DEVICE_LIMIT_FIELDS and value > spec[DEVICE_LIMIT_FIELDS[column]]:
                limit = spec[DEVICE_LIMIT_FIELDS[column]]
                issues.append(
                    _issue(
                        row_number,
                        column,
                        f"Record {row_number} {column} is {value}, above the {device_id} configuration limit of {limit}.",
                        [row_number],
                        "invalid",
                    )
                )

        air = numbers.get("Air temperature [K]")
        process = numbers.get("Process temperature [K]")
        if air is not None and process is not None and process < air:
            issues.append(
                _issue(
                    row_number,
                    "Process temperature [K]",
                    f"Record {row_number} process temperature {process} K is below air temperature {air} K.",
                    [row_number],
                    "invalid",
                )
            )

        stamp = parsed_by_row.get(row_number)
        if spec is not None and stamp is not None and not pd.isna(stamp):
            if stamp < spec["installed_at"]:
                issues.append(
                    _issue(
                        row_number,
                        "timestamp",
                        f"Record {row_number} timestamp {stamp.isoformat()} is before device {device_id} was installed at {spec['installed_at'].isoformat()}.",
                        [row_number],
                        "invalid",
                    )
                )
            if not events.empty:
                retires = events[(events["device_id"] == device_id) & (events["event_type"] == "retire")]
                if not retires.empty and stamp >= retires.iloc[0]["event_time"]:
                    retire_at = retires.iloc[0]["event_time"].isoformat()
                    issues.append(
                        _issue(
                            row_number,
                            "timestamp",
                            f"Record {row_number} timestamp {stamp.isoformat()} is at or after the retire event for {device_id} at {retire_at}.",
                            [row_number],
                            "invalid",
                        )
                    )
    for device_id, rows in unknown_devices.items():
        issues.append(
            _issue(
                None,
                "device_id",
                f"device_id {device_id!r} is not in the device configuration.",
                rows,
                "invalid",
            )
        )
    for device_id, rows in inactive_devices.items():
        status = devices[device_id]["status"]
        issues.append(
            _issue(
                None,
                "device_id",
                f"Device {device_id} is {status} and cannot be scored.",
                rows,
                "invalid",
            )
        )
    return issues


def find_issues(
    frame: pd.DataFrame,
    devices: dict,
    events: pd.DataFrame,
    *,
    missing_policy: str,
    apply_device_rules: bool,
    default_device: str = MACHINE_KEY,
) -> list[dict]:
    work = normalize_types(frame)
    if ROW_KEY not in work.columns:
        work[ROW_KEY] = np.arange(len(work))
    parsed, issues = _parsed_timestamps(work)
    order_values, _ = _order_key(work, parsed)
    issues.extend(_duplicate_issues(work, parsed, order_values))
    issues.extend(
        _value_issues(
            work,
            devices,
            events,
            parsed,
            missing_policy=missing_policy,
            apply_device_rules=apply_device_rules,
            default_device=default_device,
        )
    )
    return issues


def _order_key(frame: pd.DataFrame, parsed: pd.Series) -> tuple[pd.Series, str]:
    if parsed.notna().all() and len(parsed) > 0 and "timestamp" in frame.columns and not frame["timestamp"].map(_blank).all():
        return parsed, "timestamp"
    if "UDI" in frame.columns and frame["UDI"].map(lambda value: not _blank(value)).all():
        return frame["UDI"], "UDI"
    return frame[ROW_KEY], "input_order"


def clean_training_frame(
    frame: pd.DataFrame,
    config_path: Path | None = None,
    maintenance_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate and invalid historical rows. Keep columns otherwise unchanged."""
    devices, events, configuration_issues = load_configuration(config_path, maintenance_path)
    work = normalize_types(frame)
    work[ROW_KEY] = np.arange(len(work))
    apply_device_rules = not configuration_issues and bool(devices)
    parsed, _ = _parsed_timestamps(work)
    issues = find_issues(
        work,
        devices,
        events,
        missing_policy="allow",
        apply_device_rules=apply_device_rules,
        default_device=MACHINE_KEY,
    )
    drop_rows = {int(row) for issue in issues for row in issue.get("drop_rows", [])}
    order_values, order_by = _order_key(work, parsed)
    work = work.assign(_order=order_values)
    work = work.sort_values("_order", kind="mergesort")
    kept = work.loc[~work[ROW_KEY].isin(drop_rows)].drop(columns=[ROW_KEY, "_order"])
    if order_by == "timestamp":
        if "UDI" in kept.columns:
            kept = kept.copy()
            kept["source_udi"] = kept["UDI"]
        kept = kept.copy()
        kept["UDI"] = np.arange(1, len(kept) + 1)
    kept = kept.reset_index(drop=True)
    kind_rows = {"duplicate": set(), "invalid": set(), "timestamp": set()}
    for issue in issues:
        if issue["kind"] in kind_rows:
            kind_rows[issue["kind"]].update(int(row) for row in issue["drop_rows"])
    event_counts: dict[str, int] = {name: 0 for name in sorted(KNOWN_EVENT_TYPES)}
    if not events.empty:
        for name, count in events["event_type"].value_counts().items():
            event_counts[str(name)] = int(count)
    report = {
        "assumptions": list(ASSUMPTIONS),
        "rows_in": int(len(frame)),
        "rows_out": int(len(kept)),
        "rows_removed": int(len(frame) - len(kept)),
        "dropped": {name: len(rows) for name, rows in kind_rows.items()},
        "order_by": order_by,
        "windows_applied": bool(order_by == "timestamp" and apply_device_rules),
        "default_device_id": MACHINE_KEY,
        "device_rules_applied": bool(apply_device_rules),
        "configuration_issues": public_issues(configuration_issues),
        "maintenance_event_counts": event_counts,
        "examples": public_issues(issues)[:20],
        "devices": [
            {
                "device_id": spec["device_id"],
                "allowed_types": spec["allowed_types"],
                "status": spec["status"],
                "installed_at": spec["installed_at"].isoformat(),
            }
            for spec in devices.values()
        ],
    }
    return kept, report


def order_cycles(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Put one device's cycles in time order and number that order for lag features."""
    work = normalize_types(frame)
    if ROW_KEY not in work.columns:
        work[ROW_KEY] = np.arange(len(work))
    work["input_index"] = work[ROW_KEY].astype(int)
    parsed, _ = _parsed_timestamps(work)
    order_values, order_by = _order_key(work, parsed)
    original_positions = work["input_index"].tolist()
    work = work.assign(_order=order_values).sort_values("_order", kind="mergesort")
    reordered = work["input_index"].tolist() != original_positions
    if "UDI" in work.columns and work["UDI"].map(lambda value: not _blank(value)).any():
        work["source_udi"] = work["UDI"]
    work["UDI"] = np.arange(1, len(work) + 1)
    drop_columns = [column for column in (ROW_KEY, "_order", "device_id") if column in work.columns]
    work = work.drop(columns=drop_columns).reset_index(drop=True)
    return work, {
        "order_by": order_by,
        "reordered": bool(reordered),
        "rows": int(len(work)),
    }
