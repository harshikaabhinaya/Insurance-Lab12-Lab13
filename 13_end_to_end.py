"""LAB 13: monitoring-triggered end-to-end insurance MLOps workflow."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.common import (ROOT, RAW_DATA, INCOMING_DATA, MODEL_PATH, METADATA_PATH,
                        REPORTS, write_json, now_utc)


def run_step(label, command):
    print("\n" + "=" * 68 + f"\n{label}\n" + "=" * 68)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {label} (exit code {result.returncode})")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if not RAW_DATA.exists():
        raise FileNotFoundError("Put insurance.csv in data/raw/ or run python download_dataset.py")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stages = []
    retraining_performed = False
    batch_archive = None

    # First run: build the initial model and reference dataset.
    if not MODEL_PATH.exists() or not (ROOT / "data/reference/insurance_reference.csv").exists():
        run_step("BOOTSTRAP: INITIAL MODEL TRAINING", [sys.executable, "-m", "src.train_model"])
        stages.append({"stage": "initial_training", "status": "passed"})
        retraining_performed = True
        run_step("LAB 12: INITIAL MONITORING", [sys.executable, "12_monitoring.py"])
        stages.append({"stage": "monitoring", "status": "passed"})
    else:
        # Existing model: monitor first, then retrain only if drift and new batch are both present.
        run_step("LAB 12: MONITOR CURRENT INSURANCE DATA", [sys.executable, "12_monitoring.py"])
        stages.append({"stage": "monitoring", "status": "passed"})
        monitoring = read_json(REPORTS / "monitoring_report.json")
        drift = bool(monitoring.get("drift_detected"))
        new_batch = INCOMING_DATA.exists()
        print(f"Drift detected: {drift}")
        print(f"New incoming batch present: {new_batch}")

        if drift and new_batch:
            run_step("RETRAIN: DRIFT AND NEW DATA DETECTED", [sys.executable, "-m", "src.train_model"])
            stages.append({"stage": "retraining", "status": "passed"})
            retraining_performed = True
            archive_dir = ROOT / "data" / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            destination = archive_dir / f"insurance_batch_{stamp}.csv"
            shutil.move(str(INCOMING_DATA), str(destination))
            batch_archive = str(destination.relative_to(ROOT))
        elif drift:
            stages.append({"stage": "retraining", "status": "skipped_no_new_batch"})
            print("Drift exists, but no new batch was supplied. Retraining skipped.")
        else:
            stages.append({"stage": "retraining", "status": "skipped_no_drift"})
            print("No drift detected. Retraining skipped.")

    metadata = read_json(METADATA_PATH)
    monitoring = read_json(REPORTS / "monitoring_report.json")
    summary = {
        "pipeline_status": "passed", "completed_at": now_utc(),
        "stages": stages, "selected_model": metadata["model_name"],
        "selected_metrics": metadata["metrics"],
        "drift_detected": monitoring.get("drift_detected", False),
        "incoming_batch_present_at_run": INCOMING_DATA.exists() or batch_archive is not None,
        "retraining_performed": retraining_performed,
        "archived_batch": batch_archive,
        "model_artifact": str(MODEL_PATH.relative_to(ROOT)),
        "deployment": "Local FastAPI service is available to start with uvicorn api:app --reload. No cloud deployment is claimed."
    }
    write_json(REPORTS / "lab13_pipeline_report.json", summary)
    print("\nLAB 13 COMPLETED")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
