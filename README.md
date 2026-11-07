# Insurance ML Project — Lab 12 and Lab 13

This project extends the supplied Insurance Charges Prediction notebook using the Churn project's monitoring and MLOps workflow. It keeps the insurance target `charges` and the four regression algorithms: Linear Regression, Ridge Regression, Random Forest and Gradient Boosting.

## Project structure

```text
insurance-lab12-lab13/
├── data/
│   ├── raw/insurance.csv                 # original dataset
│   ├── reference/insurance_reference.csv # reference copy created at training
│   └── incoming/insurance_new_batch.csv # optional new batch for monitoring
├── models/                               # saved model and metadata
├── reports/                              # monitoring and pipeline reports
├── logs/                                 # API request latency log
├── src/
│   ├── common.py
│   └── train_model.py
├── download_dataset.py
├── 12_monitoring.py
├── 13_end_to_end.py
├── api.py
└── requirements.txt
```

## Step-by-step setup in Windows VS Code

### Step 1 — Extract and open
1. Extract this ZIP file.
2. Open VS Code.
3. Select **File → Open Folder**.
4. Select the extracted `insurance-lab12-lab13` folder.
5. Open **Terminal → New Terminal**.

### Step 2 — Create a virtual environment
Run these commands one at a time in the VS Code terminal:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your computer has a different Python version, replace `3.13` with the installed version. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

### Step 3 — Add the insurance dataset
The original uploaded ZIP contains the notebook and PDF, but not `insurance.csv`. Download the standard dataset by running:

```powershell
python download_dataset.py
```

If you already have your exact `insurance.csv`, copy it into `data/raw/insurance.csv` instead. Required columns are `age, sex, bmi, children, smoker, region, charges`.

### Step 4 — Run Lab 13 (initial end-to-end training)

```powershell
python 13_end_to_end.py
```

What happens:
- Checks the dataset columns and removes duplicate/incomplete rows.
- Splits the data into 80% training and 20% testing.
- Preprocesses numerical and categorical features.
- Trains all four regression models from your notebook.
- Compares MAE, RMSE and R².
- Saves the highest-R² model to `models/best_insurance_model.joblib`.
- Saves reference data, metadata and reports.
- Runs Lab 12 monitoring.
- On later runs, checks drift and retrains only when drift AND a new incoming batch are both present.

Expected output includes the four model metric rows, the selected best model and `LAB 13 COMPLETED`. Exact metric values depend on the dataset and its rows.

### Step 5 — Understand Lab 12 monitoring
Run this whenever you want a fresh monitoring report:

```powershell
python 12_monitoring.py
```

It compares the monitored data with the reference dataset:
- Numeric feature drift: Kolmogorov–Smirnov test and PSI for `age`, `bmi`, `children`.
- Categorical feature drift: chi-square test for `sex`, `smoker`, `region`.
- Prediction drift: compares the distribution of predicted charges using the KS test.
- Model performance: calculates MAE, RMSE and R² when actual `charges` are available.
- API performance: calculates request count, average latency, p95 latency and maximum latency from API logs, when requests have been made.

Reports are saved in `reports/data_drift_report.csv` and `reports/monitoring_report.json`.

### Step 6 — Test with a new incoming batch
1. Create a CSV with the same seven required columns.
2. Save it as `data/incoming/insurance_new_batch.csv`.
3. Run `python 12_monitoring.py` to check for drift.
4. Run `python 13_end_to_end.py`. The Lab 13 workflow checks the monitoring report and retrains only when BOTH drift and a new incoming batch are present. If there is no drift, or no new batch, retraining is skipped. After successful ingestion, the incoming batch is moved to `data/archive/` to prevent repeated ingestion.

For a valid performance evaluation, the incoming batch should contain real `charges` and should be a separate holdout batch that was not used to train the model. A synthetic batch can demonstrate drift detection but should not be described as real-world evidence.

### Step 7 — Start the prediction API
Keep the VS Code terminal in the project folder and run:

```powershell
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser. Expand **POST /predict**, click **Try it out**, and use this sample:

```json
{
  "age": 25,
  "sex": "female",
  "bmi": 28.5,
  "children": 1,
  "smoker": "no",
  "region": "southeast"
}
```

Click **Execute**. The response contains `predicted_insurance_charges`. Each API request is logged in `logs/api_requests.jsonl`.

### Step 8 — Monitor API performance
After making several API requests, stop the API with Ctrl+C and run:

```powershell
python 12_monitoring.py
```

The monitoring JSON now includes API request count, average latency, p95 latency and maximum latency.

## Lab 12 and Lab 13 outputs

| Output | Meaning |
|---|---|
| `reports/data_drift_report.csv` | Feature-by-feature drift results |
| `reports/monitoring_report.json` | Data drift, prediction drift, regression metrics and API latency |
| `reports/model_comparison.csv` | Comparison of the four regression models |
| `reports/training_report.json` | Best model and training metrics |
| `reports/lab13_pipeline_report.json` | End-to-end pipeline execution summary |
| `models/best_insurance_model.joblib` | Trained prediction pipeline |
| `models/model_metadata.json` | Selected model, metrics and training information |

## Important academic notes

- R² is a regression goodness-of-fit metric. Do not label `R² × 100` as conventional classification accuracy.
- Data drift is a change in input feature distributions; prediction drift is a change in predicted-charge distributions.
- A drift alert does not prove that model quality has declined. Use a separate labelled holdout batch to measure MAE, RMSE and R².
- This is a local MLOps demonstration. It does not claim cloud hosting, Kubernetes deployment, scheduled production monitoring or a live insurance-company data feed.
