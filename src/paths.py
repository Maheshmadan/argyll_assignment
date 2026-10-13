"""Project paths shared by training and the deployment API."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ai4i2020.csv"
DATA_DIR = ROOT / "data"
DEVICE_CONFIG_PATH = DATA_DIR / "device_configurations.csv"
MAINTENANCE_EVENTS_PATH = DATA_DIR / "maintenance_events.csv"
SENSOR_READINGS_PATH = DATA_DIR / "sensor_readings.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
MLRUNS_DIR = ROOT / "mlruns"


def tracking_uri() -> str:
    """SQLite tracking store. The file backend is deprecated in current MLflow."""
    db_path = (ROOT / "mlflow.db").resolve()
    return "sqlite:///" + db_path.as_posix()
