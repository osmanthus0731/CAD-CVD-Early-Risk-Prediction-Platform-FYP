@echo off
setlocal

if not exist "data\combined\cad_risk_model.json" (
  echo Training CAD risk model first...
  python models\train_cad_risk_model.py
  if errorlevel 1 (
    echo Model training failed.
    exit /b 1
  )
)

echo Starting CAD risk app on http://127.0.0.1:8000
python app\cad_risk_app.py
