from pathlib import Path
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "insurance.csv"
REFERENCE_DATA = ROOT / "data" / "reference" / "insurance_reference.csv"
INCOMING_DATA = ROOT / "data" / "incoming" / "insurance_new_batch.csv"
MODEL_PATH = ROOT / "models" / "best_insurance_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, default=str), encoding="utf-8")


def load_insurance_csv(path):
    import pandas as pd
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place insurance.csv inside data/raw/. The project also contains a dataset download helper."
        )
    df = pd.read_csv(path)
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[EXPECTED_COLUMNS].copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=EXPECTED_COLUMNS)
    df[NUMERIC_FEATURES + ["charges"]] = df[NUMERIC_FEATURES + ["charges"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=NUMERIC_FEATURES + ["charges"])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower()
    if len(df) < 20:
        raise ValueError("At least 20 valid rows are required to train and monitor the model.")
    return df.reset_index(drop=True)
