import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common import (ROOT, RAW_DATA, INCOMING_DATA, REFERENCE_DATA, MODEL_PATH,
                    METADATA_PATH, REPORTS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                    load_insurance_csv, write_json, now_utc)


def train():
    df = load_insurance_csv(RAW_DATA)
    # Append incoming batch only when a valid new batch is present.
    if INCOMING_DATA.exists():
        incoming = load_insurance_csv(INCOMING_DATA)
        df = pd.concat([df, incoming], ignore_index=True).drop_duplicates().reset_index(drop=True)
        print(f"Incoming rows included: {len(incoming)}")

    X, y = df.drop(columns="charges"), df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ])
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {
            "model": name,
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred))
        }
        rows.append(row)
        fitted[name] = pipe
        print(f"{name}: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, R2={row['r2']:.4f}")

    # Select highest R2; use lower MAE as tie-breaker.
    rows.sort(key=lambda r: (r["r2"], -r["mae"]), reverse=True)
    best = rows[0]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best["model"]], MODEL_PATH)
    REFERENCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REFERENCE_DATA, index=False)
    metadata = {
        "model_name": best["model"], "metrics": best,
        "all_model_metrics": rows, "training_rows": len(df),
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": "charges", "trained_at": now_utc()
    }
    write_json(METADATA_PATH, metadata)
    pd.DataFrame(rows).to_csv(REPORTS / "model_comparison.csv", index=False)
    write_json(REPORTS / "training_report.json", metadata)
    print(f"\nBest model saved: {MODEL_PATH}")
    print(f"Best model: {best['model']} | R2: {best['r2']:.4f}")
    return metadata

if __name__ == "__main__":
    train()
