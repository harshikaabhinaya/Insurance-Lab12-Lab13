"""LAB 12: data drift, prediction drift, regression performance and API latency monitoring."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common import (RAW_DATA, REFERENCE_DATA, INCOMING_DATA, MODEL_PATH,
                        METADATA_PATH, REPORTS, LOGS, NUMERIC_FEATURES,
                        CATEGORICAL_FEATURES, load_insurance_csv, write_json, now_utc)


def population_stability_index(reference, current, bins=10):
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    if not MODEL_PATH.exists() or not REFERENCE_DATA.exists():
        raise FileNotFoundError("Train the model first: run python run_pipeline.py")
    reference = load_insurance_csv(REFERENCE_DATA)
    source = INCOMING_DATA if INCOMING_DATA.exists() else RAW_DATA
    current = load_insurance_csv(source)
    drift_rows = []

    for col in NUMERIC_FEATURES:
        ks = float(ks_2samp(reference[col], current[col]).statistic)
        p_value = float(ks_2samp(reference[col], current[col]).pvalue)
        psi = population_stability_index(reference[col], current[col])
        drift = bool(p_value < 0.05 or psi >= 0.20)
        drift_rows.append({"feature": col, "type": "numeric", "ks_statistic": ks,
                           "p_value": p_value, "psi": psi, "status": "DRIFT" if drift else "STABLE"})

    for col in CATEGORICAL_FEATURES:
        ref_counts = reference[col].value_counts(normalize=True)
        cur_counts = current[col].value_counts(normalize=True)
        categories = sorted(set(ref_counts.index) | set(cur_counts.index))
        observed = np.array([[reference[col].value_counts().get(k, 0) for k in categories],
                             [current[col].value_counts().get(k, 0) for k in categories]])
        # Chi-square requires at least two categories and nonzero expected frequencies.
        if len(categories) > 1 and observed.sum() > 0:
            try:
                p_value = float(chi2_contingency(observed)[1])
            except ValueError:
                p_value = 1.0
        else:
            p_value = 1.0
        drift = p_value < 0.05
        drift_rows.append({"feature": col, "type": "categorical", "p_value": p_value,
                           "status": "DRIFT" if drift else "STABLE"})

    model = joblib.load(MODEL_PATH)
    reference_pred = model.predict(reference.drop(columns="charges"))
    current_pred = model.predict(current.drop(columns="charges"))
    prediction_ks = ks_2samp(reference_pred, current_pred)
    prediction_drift = bool(prediction_ks.pvalue < 0.05)

    performance = None
    if "charges" in current.columns:
        performance = {
            "mae": float(mean_absolute_error(current["charges"], current_pred)),
            "rmse": float(np.sqrt(mean_squared_error(current["charges"], current_pred))),
            "r2": float(r2_score(current["charges"], current_pred)),
            "evaluated_rows": int(len(current)),
            "note": "Metrics are meaningful as a live holdout only when this batch was not used for model training."
        }

    api_log = LOGS / "api_requests.jsonl"
    latency_summary = None
    if api_log.exists():
        entries = [json.loads(line) for line in api_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if entries:
            values = [e["latency_ms"] for e in entries]
            latency_summary = {"request_count": len(values), "average_ms": float(np.mean(values)),
                               "p95_ms": float(np.percentile(values, 95)), "max_ms": float(np.max(values))}

    drift_detected = any(r["status"] == "DRIFT" for r in drift_rows) or prediction_drift
    report = {
        "timestamp": now_utc(), "source_dataset": str(source), "rows_monitored": len(current),
        "data_drift_detected": any(r["status"] == "DRIFT" for r in drift_rows),
        "prediction_drift_detected": prediction_drift, "drift_detected": drift_detected,
        "feature_drift": drift_rows,
        "prediction_drift": {"ks_statistic": float(prediction_ks.statistic),
                              "p_value": float(prediction_ks.pvalue),
                              "status": "DRIFT" if prediction_drift else "STABLE"},
        "performance": performance, "api_latency": latency_summary
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(drift_rows).to_csv(REPORTS / "data_drift_report.csv", index=False)
    write_json(REPORTS / "monitoring_report.json", report)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
