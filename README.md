# CAD Risk Prediction Project

This project builds and runs a local Coronary Artery Disease/Cardiovascular Disease risk prediction web app. It combines multiple health datasets, trains CAD/CVD risk models, compares the project score with Framingham and SCORE2 benchmark equations, and serves a browser-based risk form.

The app is for educational screening support only. It is not a medical diagnosis tool.

## Project Structure

```text
app/                 Local HTTP server and prediction API
webapp/              Browser UI files
preprocessing/       Dataset cleaning and combined-dataset scripts
models/              Model training and baseline scripts
evaluation/          Model and benchmark comparison scripts
data/                Raw, cleaned, processed, combined, and output data
catboost_info/       Training logs from CatBoost experiments
run_cad_app.bat      Windows shortcut to train the model if needed and start the app
```

## Requirements

- Python 3.10 or newer
- Windows PowerShell or Command Prompt

Install the main Python packages:

```powershell
python -m pip install numpy pandas scikit-learn
```

Optional packages used only by the stronger model comparison script:

```powershell
python -m pip install xgboost lightgbm catboost
```

## Quick Start

From the project folder:

```powershell
cd "C:\Users\your_name\Downloads\Final year project"
```

Start the app with the batch file:

```powershell
.\run_cad_app.bat
```

Then open this URL in your browser:

```text
http://127.0.0.1:8000
```

The batch file checks for `data/combined/cad_risk_model.json`. If the file is missing, it trains the lightweight CAD risk model first, then starts the web app.

To stop the app, press `Ctrl + C` in the terminal.

## Run Manually

If you prefer to run each step yourself, use this order.

### 1. Build Combined Dataset

```powershell
python preprocessing\build_cad_combined_dataset.py
```

Output:

```text
data/combined/cad_risk_training.csv
data/combined/cad_risk_summary.json
```

### 2. Train Lightweight App Model

```powershell
python models\train_cad_risk_model.py
```

Output:

```text
data/combined/cad_risk_model.json
data/combined/cad_risk_model_metrics.json
```

### 3. Train AUC-Optimized Model

```powershell
python models\train_auc_optimized_model.py
```

Output:

```text
data/combined/cad_auc_model.pkl
data/combined/cad_auc_model_metrics.json
```

The app automatically uses `cad_auc_model.pkl` when it exists. If it does not exist, the app still runs using the transparent clinical-style score and the lightweight model files.

### 4. Generate Benchmark Comparison

```powershell
python evaluation\compare_risk_scores.py
```

Output:

```text
data/combined/risk_score_comparison.json
```

This compares the project model with Framingham and SCORE2 on eligible rows from the processed CVD dataset.

### 5. Start the Web App

```powershell
python app\cad_risk_app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Deploy to Vercel

Import the whole project with the directory containing `vercel.json` as the
**Root Directory** (do not select `webapp` or `app`). Use **Other** as the
Framework Preset. Clear any custom Build Command and Install Command overrides;
the checked-in configuration serves `webapp` and installs `requirements.txt`.
The Output Directory is `webapp`.

`api/index.py` exposes the existing Python request handler as a Vercel Function.
The `/api/*` routes are forwarded to it, while HTML, CSS, and JavaScript are
served as static files. Vercel does not run `run_cad_app.bat` or start the local
HTTP server.

Include the saved files `data/combined/cad_risk_model.json`,
`data/combined/cad_auc_model.pkl`, and
`data/combined/risk_score_comparison.json` in the repository you deploy.
The first file is required; the other two provide the stronger model and
benchmark comparison. Training does not run during deployment. The
`.vercelignore` file excludes datasets, reports, and training scripts from uploads.

After redeploying, open `/api/health` on your deployment URL, then submit the
prediction form. A successful health response has `"status": "ok"`.
Python 3.12 and the runtime package versions are specified in the project;
scikit-learn matches the version used to save the bundled model.

## Other Useful Commands

Run the NumPy KNN baseline:

```powershell
python models\knn_cad_baseline.py
```

Compare stronger scikit-learn models:

```powershell
python evaluation\compare_stronger_models.py
```

Clean/preprocess all CSV files:

```powershell
python preprocessing\preprocess_all_csv.py
```

Preprocess all supported data files:

```powershell
python preprocessing\preprocess_all_data.py
```

## App API Endpoints

When the app is running:

- `GET /` serves the web interface.
- `GET /api/health` checks that the server and model are loaded.
- `GET /api/model-info` returns model fields and metadata.
- `GET /api/comparison` returns benchmark comparison results if generated.
- `POST /api/predict` returns a CAD/CVD risk prediction for form input.

## Notes

- Run commands from the project root so relative paths resolve correctly.
- Generated model and metric files are stored under `data/combined/`.
- If `risk_score_comparison.json` is missing, the app still runs, but the benchmark comparison view will ask you to run `python evaluation\compare_risk_scores.py`.
- The published Framingham and SCORE2 comparisons are directional benchmarks because the local datasets do not represent formal adjudicated 10-year clinical outcome cohorts.
