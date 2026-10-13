"""Data-quality rules for historical files and live requests."""

from pathlib import Path

import pandas as pd

from src.monitoring import psi_status
from src.paths import DATA_PATH, SENSOR_READINGS_PATH
from src.preprocess import load_raw, prepare_training_frame
from src.quality import clean_training_frame


def cycle(**overrides):
    row = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0,
        "Product ID": "L100",
        "UDI": 1,
        "timestamp": "2020-02-01T00:00:00Z",
        "device_id": "MILL_01",
    }
    row.update(overrides)
    return row


def test_published_dataset_is_already_clean():
    raw = load_raw(DATA_PATH)
    cleaned, report = clean_training_frame(raw)
    assert report["rows_removed"] == 0
    assert report["device_rules_applied"] is True
    assert report["order_by"] == "UDI"
    assert list(cleaned.columns) == list(raw.columns)
    assert cleaned["UDI"].tolist() == raw["UDI"].tolist()
    assert cleaned["Torque [Nm]"].tolist() == raw["Torque [Nm]"].tolist()
    assert report["maintenance_event_counts"]["tool_change"] == 1
    assert report["maintenance_event_counts"]["retire"] == 1
    assert report["maintenance_event_counts"]["install"] == 2


def test_duplicate_invalid_and_config_rows_are_removed():
    frame = pd.DataFrame(
        [
            cycle(UDI=1, timestamp="2020-03-02T00:00:00Z", **{"Product ID": "L1"}),
            cycle(UDI=1, timestamp="2020-03-01T00:00:00Z", **{"Product ID": "L2"}),
            cycle(UDI=3, timestamp="2020-03-03T00:00:00Z", **{"Product ID": "L3", "Torque [Nm]": -5}),
            cycle(UDI=4, timestamp="2020-03-04T00:00:00Z", **{"Product ID": "M4", "Type": "L"}),
            cycle(UDI=5, timestamp="2019-01-01T00:00:00Z", **{"Product ID": "L5"}),
            cycle(UDI=6, timestamp="2020-08-01T00:00:00Z", device_id="MILL_02", Type="M", **{"Product ID": "M6"}),
            cycle(
                UDI=7,
                timestamp="2020-08-02T00:00:00Z",
                device_id="MILL_02",
                Type="L",
                **{"Product ID": "L7", "Rotational speed [rpm]": 5000},
            ),
            cycle(UDI=8, timestamp="2021-06-01T00:00:00Z", device_id="MILL_02", Type="L", **{"Product ID": "L8"}),
            cycle(UDI=9, timestamp="2020-05-01T00:00:00Z", **{"Product ID": "L9"}),
        ]
    )
    cleaned, report = clean_training_frame(frame)
    assert report["rows_removed"] == 7
    assert cleaned["source_udi"].tolist() == [1, 9]
    assert any("outside the accepted range" in item["message"] for item in report["examples"])
    assert any("does not start with Type" in item["message"] for item in report["examples"])
    assert any("was installed" in item["message"] for item in report["examples"])
    assert any("not configured to run" in item["message"] for item in report["examples"])
    assert any("configuration limit" in item["message"] for item in report["examples"])
    assert any("retire event" in item["message"] for item in report["examples"])
    assert any("share UDI" in item["message"] for item in report["examples"])


def test_duplicate_keeps_the_earlier_timestamp():
    frame = pd.DataFrame(
        [
            cycle(UDI=2, timestamp="2020-04-01T00:00:00Z", **{"Product ID": "L50"}),
            cycle(UDI=1, timestamp="2020-03-01T00:00:00Z", **{"Product ID": "L50"}),
        ]
    )
    cleaned, report = clean_training_frame(frame)
    assert report["rows_removed"] == 1
    assert cleaned["source_udi"].tolist() == [1]


def test_unparseable_timestamp_is_removed():
    frame = pd.DataFrame([cycle(timestamp="yesterday"), cycle(UDI=2, **{"Product ID": "L101"})])
    cleaned, report = clean_training_frame(frame)
    assert report["rows_removed"] == 1
    assert any("ISO-8601" in item["message"] for item in report["examples"])
    assert len(cleaned) == 1


def test_process_temperature_below_air_is_removed():
    frame = pd.DataFrame([cycle(**{"Process temperature [K]": 290.0})])
    cleaned, report = clean_training_frame(frame)
    assert len(cleaned) == 0
    assert any("below air temperature" in item["message"] for item in report["examples"])


def test_missing_config_does_not_drop_physical_rows(tmp_path: Path):
    frame = pd.DataFrame(
        [
            cycle(),
            cycle(UDI=2, timestamp="2020-02-02T00:00:00Z", **{"Product ID": "L101"}),
        ]
    )
    cleaned, report = clean_training_frame(
        frame,
        config_path=tmp_path / "missing.csv",
        maintenance_path=tmp_path / "missing_events.csv",
    )
    assert report["rows_removed"] == 0
    assert report["device_rules_applied"] is False
    assert report["configuration_issues"]


def test_prepare_training_frame_records_quality(tmp_path: Path):
    raw = load_raw(DATA_PATH).head(60)
    subset = tmp_path / "subset.csv"
    raw.to_csv(subset, index=False)
    _features, _target, report = prepare_training_frame(subset)
    assert report["quality"]["rows_removed"] == 0
    assert report["rows_after_lag_warmup"] == 57


def test_sensor_sample_passes_the_same_rules():
    sample = pd.read_csv(SENSOR_READINGS_PATH)
    cleaned, report = clean_training_frame(sample)
    assert report["rows_removed"] == 0
    assert len(cleaned) == len(sample)


def test_retired_device_and_conflicting_install_are_reported(tmp_path: Path):
    header = "device_id,allowed_types,status,process,installed_at,max_rotational_speed_rpm,max_torque_nm,max_tool_wear_min\n"
    events = "device_id,event_time,event_type,component,notes\n"
    retired_config = tmp_path / "retired.csv"
    retired_events = tmp_path / "retired_events.csv"
    retired_config.write_text(header + "MILL_09,L,retired,milling,2020-01-01T00:00:00Z,10000,500,10000\n", encoding="utf-8")
    retired_events.write_text(events + "MILL_09,2020-01-01T00:00:00Z,install,line,installed\n", encoding="utf-8")
    cleaned, report = clean_training_frame(
        pd.DataFrame([cycle(device_id="MILL_09")]),
        retired_config,
        retired_events,
    )
    assert cleaned.empty
    assert any("cannot be scored" in item["message"] for item in report["examples"])

    conflict_config = tmp_path / "conflict.csv"
    conflict_events = tmp_path / "conflict_events.csv"
    conflict_config.write_text(header + "MILL_01,\"L,M,H\",active,milling,2020-01-01T00:00:00Z,10000,500,10000\n", encoding="utf-8")
    conflict_events.write_text(events + "MILL_01,2020-06-01T00:00:00Z,install,line,later than the configuration\n", encoding="utf-8")
    cleaned, report = clean_training_frame(pd.DataFrame([cycle()]), conflict_config, conflict_events)
    assert report["device_rules_applied"] is False
    assert report["rows_removed"] == 0
    assert any("does not match installed_at" in item["message"] for item in report["configuration_issues"])


def test_psi_bands():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.2) == "moderate_shift"
    assert psi_status(0.4) == "major_shift"
