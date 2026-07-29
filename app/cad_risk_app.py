# interface to use ml model

from __future__ import annotations

import json
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "combined" / "cad_risk_model.json"
AUC_MODEL_PATH = ROOT / "data" / "combined" / "cad_auc_model.pkl"
COMPARISON_PATH = ROOT / "data" / "combined" / "risk_score_comparison.json"
STATIC_DIR = ROOT / "webapp"
MMOL_PER_MGDL_CHOLESTEROL = 1.0 / 38.67

SCORE2_COEFFICIENTS = {
    "male": {
        "age": 0.3742,
        "smoking": 0.6012,
        "sbp": 0.2777,
        "diabetes": 0.6457,
        "tchol": 0.1458,
        "hdl": -0.2698,
        "age_smoking": -0.0755,
        "age_sbp": -0.0255,
        "age_tchol": -0.0281,
        "age_hdl": 0.0426,
        "age_diabetes": -0.0983,
        "baseline_survival": 0.9605,
        "scale_1": {"low": -0.5699},
        "scale_2": {"low": 0.7476},
    },
    "female": {
        "age": 0.4648,
        "smoking": 0.7744,
        "sbp": 0.3131,
        "diabetes": 0.8096,
        "tchol": 0.1002,
        "hdl": -0.2606,
        "age_smoking": -0.1088,
        "age_sbp": -0.0277,
        "age_tchol": -0.0226,
        "age_hdl": 0.0613,
        "age_diabetes": -0.1272,
        "baseline_survival": 0.9776,
        "scale_1": {"low": -0.7380},
        "scale_2": {"low": 0.7019},
    },
}

FRAMINGHAM_LIPIDS = {
    "male": {
        "age": 3.06117,
        "tchol": 1.12370,
        "hdl": -0.93263,
        "sbp_untreated": 1.93303,
        "smoking": 0.65451,
        "diabetes": 0.57367,
        "baseline_survival": 0.88936,
        "mean_sum": 23.9802,
    },
    "female": {
        "age": 2.32888,
        "tchol": 1.20904,
        "hdl": -0.70833,
        "sbp_untreated": 2.76157,
        "smoking": 0.52873,
        "diabetes": 0.69154,
        "baseline_survival": 0.95012,
        "mean_sum": 26.1931,
    },
}


# Keep very large or small numbers stable before converting to probability.
def sigmoid(value: float) -> float:
    value = float(np.clip(value, -35.0, 35.0))
    return 1.0 / (1.0 + np.exp(-value))


# Read the saved model settings and weights from the JSON file.
def load_model():
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


MODEL = load_model()


# Load the optional AUC-optimized model from pickle.
def load_auc_model():
    # Load the stronger sklearn model if it has been trained.
    if not AUC_MODEL_PATH.exists():
        return None
    with AUC_MODEL_PATH.open("rb") as f:
        return pickle.load(f)


AUC_MODEL = load_auc_model()


# Load saved benchmark comparison results for the app.
def load_comparison_summary():
    # Read the benchmark metrics shown in the "Compare benchmarks" modal.
    if not COMPARISON_PATH.exists():
        return {
            "available": False,
            "message": "Comparison file not found. Run python evaluation/compare_risk_scores.py to generate benchmark results.",
        }

    data = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    framingham = data["framingham_comparison"]
    score2 = data["score2_comparison"]
    return {
        "available": True,
        "comparison_date": data.get("comparison_date"),
        "notes": data.get("clinical_alignment_notes", []),
        "references": data.get("source_references", {}),
        "benchmarks": [
            {
                "name": "Framingham",
                "eligibility": framingham["eligibility"],
                "project": framingham["project_model"],
                "published": framingham["framingham_lipids"],
                "published_label": "Framingham lipid score",
            },
            {
                "name": "SCORE2",
                "eligibility": score2["eligibility"],
                "project": score2["project_model"],
                "published": score2["score2_low_risk_region"],
                "published_label": "SCORE2 low-risk region",
            },
        ],
        "interpretation": (
            "Project metrics are higher than both Framingham and SCORE2, with lower error in this scenario. "
            "This project has better ranking on the dataset and outcome overall; however, it should not be used in deployment "
            "as it is not verified to be superior to any model. Framingham and SCORE2 were designed for specific clinical "
            "cohorts and 10-year event outcomes, not individual results."
        ),
    }

FASTING_GLUCOSE_FIELD = "fasting_glucose"
FASTING_GLUCOSE_NOTE = (
    "Fasting glucose is optional. When it is blank, the app estimates a neutral glucose category "
    "from the available answers instead of treating the missing value as a risk factor."
)
INPUT_RANGES = {
    "age_years": (18, 100, "Age must be between 18 and 100 years."),
    "bmi": (10, 70, "BMI must be between 10 and 70."),
    "systolic_bp": (70, 240, "Systolic BP must be between 70 and 240 mmHg."),
    "diastolic_bp": (40, 140, "Diastolic BP must be between 40 and 140 mmHg."),
    "cholesterol": (80, 400, "Total cholesterol must be between 80 and 400 mg/dL."),
    "fasting_glucose": (50, 300, "Fasting glucose must be between 50 and 300 mg/dL."),
}


# Convert a user input value into a number when possible.
def parse_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# Check whether numeric inputs are inside realistic ranges.
def validate_payload(payload: dict) -> list[str]:
    # Stop unrealistic form values before they reach the model.
    errors = []
    for field, (minimum, maximum, message) in INPUT_RANGES.items():
        parsed = parse_number(payload.get(field))
        if parsed is None:
            continue
        if parsed < minimum or parsed > maximum:
            errors.append(message)
    return errors


# Convert yes/no and sex fields into binary 
def parse_binary(field: str, value):
    # Convert dropdown answers and yes/no text into numeric model values.
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if field == "sex":
        if text in {"male", "m", "1", "1.0"}:
            return 1.0
        if text in {"female", "f", "0", "2", "2.0"}:
            return 0.0
        return None
    if text in {"yes", "y", "true", "1", "1.0"}:
        return 1.0
    if text in {"no", "n", "false", "0", "0.0"}:
        return 0.0
    return parse_number(value)


# Convert fasting glucose input into the category scale used by the model.
def glucose_to_model_scale(value: float) -> float:
    # Training data mixes glucose categories and mg/dL, so map both to one scale.
    if value < 40:
        return float(np.clip(value, 0.0, 4.0))
    if value < 100:
        return 1.0
    if value < 126:
        return 2.0
    if value < 180:
        return 3.0
    return 4.0


# Estimate a glucose category when fasting glucose is left blank.
def estimate_glucose_scale(parsed_values: dict[str, float | None]) -> float:
    # If glucose is blank, use a neutral estimate based on related answers.
    diabetes = parsed_values.get("diabetes")
    bmi = parsed_values.get("bmi")
    age = parsed_values.get("age_years")

    estimate = 1.0
    if diabetes is not None and diabetes >= 0.5:
        estimate = 3.0
    elif bmi is not None and bmi >= 30:
        estimate = 2.0
    elif age is not None and age >= 60:
        estimate = 2.0
    return estimate


# Create text that explains an estimated glucose category.
def glucose_display_label(value: float) -> str:
    if value <= 1:
        return "estimated normal"
    if value <= 2:
        return "estimated borderline"
    if value <= 3:
        return "estimated high"
    return "estimated very high"


# Build extra features for the AUC-optimized model.
def make_auc_engineered_features(x_filled: np.ndarray) -> np.ndarray:
    # Engineered features used by the AUC-optimized sklearn model.
    age = x_filled[..., 0]
    bmi = x_filled[..., 2]
    systolic_bp = x_filled[..., 3]
    diastolic_bp = x_filled[..., 4]
    cholesterol = x_filled[..., 5]
    hdl = np.maximum(x_filled[..., 6], 1.0)
    ldl = x_filled[..., 7]
    triglycerides = x_filled[..., 8]
    smoking = x_filled[..., 10]
    diabetes = x_filled[..., 11]
    family_history = x_filled[..., 12]

    return np.array(
        [
            systolic_bp - diastolic_bp,
            (systolic_bp + (2.0 * diastolic_bp)) / 3.0,
            cholesterol - hdl,
            cholesterol / hdl,
            ldl / hdl,
            triglycerides / hdl,
            age * systolic_bp,
            age * cholesterol,
            bmi * systolic_bp,
            smoking * age,
            diabetes * age,
            family_history * age,
        ],
        dtype=float,
    )


# Predict CAD risk using the optional AUC-optimized model.
def auc_model_probability(payload: dict) -> float | None:
    # Predict with the stronger model when the pickle file is available.
    if AUC_MODEL is None:
        return None

    fields = AUC_MODEL["fields"]
    numeric_fields = set(AUC_MODEL["numeric_fields"])
    binary_fields = set(AUC_MODEL["binary_fields"])
    raw_values = []
    for field in fields:
        value = payload.get(field)
        parsed = parse_number(value) if field in numeric_fields else parse_binary(field, value) if field in binary_fields else None
        raw_values.append(np.nan if parsed is None else float(parsed))

    x_raw = np.array(raw_values, dtype=float)
    medians = np.asarray(AUC_MODEL["medians"], dtype=float)
    x_filled = np.where(np.isnan(x_raw), medians, x_raw)
    x_full = np.concatenate([x_filled, make_auc_engineered_features(x_filled), np.isnan(x_raw).astype(float)])
    probability = float(AUC_MODEL["model"].predict_proba([x_full])[0][1])
    return float(np.clip(probability, 0.0, 1.0))


# Create extra calculated features from the original patient inputs.
def make_engineered_features(x_filled: np.ndarray) -> np.ndarray:
    age = x_filled[..., 0]
    bmi = x_filled[..., 2]
    systolic_bp = x_filled[..., 3]
    diastolic_bp = x_filled[..., 4]
    cholesterol = x_filled[..., 5]
    fasting_glucose = x_filled[..., 6]
    smoking = x_filled[..., 7]
    diabetes = x_filled[..., 8]
    family_history = x_filled[..., 9]

    pulse_pressure = systolic_bp - diastolic_bp
    mean_arterial_pressure = (systolic_bp + (2.0 * diastolic_bp)) / 3.0
    age_x_systolic_bp = age * systolic_bp
    age_x_cholesterol = age * cholesterol
    bmi_x_systolic_bp = bmi * systolic_bp
    cholesterol_x_glucose = cholesterol * fasting_glucose
    smoking_x_age = smoking * age
    diabetes_x_age = diabetes * age
    family_history_x_age = family_history * age

    return np.array(
        [
            pulse_pressure,
            mean_arterial_pressure,
            age_x_systolic_bp,
            age_x_cholesterol,
            bmi_x_systolic_bp,
            cholesterol_x_glucose,
            smoking_x_age,
            diabetes_x_age,
            family_history_x_age,
        ],
        dtype=float,
    )



#PREPROCESS USER INPUTS#
# Prepare form data so it matches the same format used during model training.
def preprocess_payload(payload: dict):
    # Convert browser form data into the exact feature layout expected by the model.
    fields = MODEL["question_fields"]
    numeric_fields = set(MODEL["numeric_fields"])
    binary_fields = set(MODEL["binary_fields"])

    parsed_values = {}
    raw_values = []
    missing_fields = []
    estimated_fields = {}
    display_overrides = {}
    for field in fields:
        value = payload.get(field)
        parsed = parse_number(value) if field in numeric_fields else parse_binary(field, value) if field in binary_fields else None
        parsed_values[field] = parsed
        if parsed is None:
            missing_fields.append(field)
            raw_values.append(np.nan)
        else:
            if field == FASTING_GLUCOSE_FIELD:
                display_overrides[field] = round(float(parsed), 3)
                raw_values.append(glucose_to_model_scale(float(parsed)))
            else:
                raw_values.append(float(parsed))

    if FASTING_GLUCOSE_FIELD in fields:
        # Fasting glucose is optional, so do not punish the user for leaving it blank.
        glucose_idx = fields.index(FASTING_GLUCOSE_FIELD)
        if np.isnan(raw_values[glucose_idx]):
            estimated_glucose = estimate_glucose_scale(parsed_values)
            raw_values[glucose_idx] = estimated_glucose
            estimated_fields[FASTING_GLUCOSE_FIELD] = glucose_display_label(estimated_glucose)
            display_overrides[FASTING_GLUCOSE_FIELD] = estimated_fields[FASTING_GLUCOSE_FIELD]
            if FASTING_GLUCOSE_FIELD in missing_fields:
                missing_fields.remove(FASTING_GLUCOSE_FIELD)

    x_raw = np.array(raw_values, dtype=float)
    medians = np.array(MODEL["medians"], dtype=float)
    means = np.array(MODEL["means"], dtype=float)
    stds = np.array(MODEL["stds"], dtype=float)
    missing_flags = np.isnan(x_raw).astype(float)
    x_filled = np.where(np.isnan(x_raw), medians, x_raw)
    engineered = make_engineered_features(x_filled)
    x_full = np.concatenate([x_filled, engineered, missing_flags])
    transformed = (x_full - means) / stds
    return transformed, missing_fields, estimated_fields, display_overrides, x_filled, x_full



#EXPLAIN MODEL FACTORS#
# Show the strongest factors that affected the prediction result.
def explain_prediction(transformed: np.ndarray, x_full: np.ndarray, display_overrides: dict | None = None):
    feature_names = MODEL["feature_names_after_transform"]
    weights = np.array(MODEL["weights"], dtype=float)
    contributions = transformed * weights
    descriptions = MODEL["feature_descriptions"]
    details = []

    question_fields = set(MODEL["question_fields"])
    for field, contribution, value_used in zip(feature_names, contributions, x_full):
        if field.endswith("_missing"):
            continue
        display_value = display_overrides.get(field) if display_overrides else None
        if display_value is None:
            display_value = round(float(value_used), 3)
        if field.endswith("_missing"):
            display_value = int(round(float(value_used)))
        details.append(
            {
                "field": field,
                "label": descriptions.get(field, field),
                "value_used": display_value,
                "contribution": round(float(contribution), 4),
                "derived": field not in question_fields,
            }
        )

    details.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return details[:5]


# Create one readable sentence explaining the risk level.
def risk_reason(top_factors: list[dict], risk_level: str) -> str:
    # Turn the top factor list into one short explanation sentence.
    increasing = [item["label"] for item in top_factors if item["contribution"] > 0]
    if increasing:
        main_factors = ", ".join(increasing[:3])
        return f"The {risk_level.lower()} percentage is mainly driven by: {main_factors}."
    return f"The {risk_level.lower()} percentage is based on how the entered values compare with the model's training patterns."


# Create one standardized risk-factor component dictionary.
def component(field: str, label: str, value, points: float, detail: str) -> dict:
    # Standard shape for one explanation card in the result page.
    return {
        "field": field,
        "label": label,
        "value_used": value,
        "contribution": round(float(points), 2),
        "derived": False,
        "effect": detail,
    }


# Give age a point value for the transparent clinical score.
def age_points(age: float) -> float:
    if age < 30:
        return 3
    if age < 40:
        return 8
    if age < 50:
        return 16
    if age < 60:
        return 26
    if age < 70:
        return 38
    return 50


# Give blood pressure a point value and explanation label.
def bp_points(systolic: float, diastolic: float) -> tuple[float, str]:
    if systolic >= 140 or diastolic >= 90:
        return 24, "stage 2 high blood pressure range"
    if systolic >= 130 or diastolic >= 80:
        return 14, "stage 1 high blood pressure range"
    if systolic >= 120:
        return 6, "elevated blood pressure range"
    return 0, "normal blood pressure range"


# Give cholesterol a point value and explanation label.
def cholesterol_points(cholesterol: float) -> tuple[float, str]:
    if cholesterol >= 240:
        return 18, "high total cholesterol range"
    if cholesterol >= 200:
        return 10, "borderline total cholesterol range"
    return 0, "desirable total cholesterol range"


# Give BMI a point value and explanation label.
def bmi_points(bmi: float) -> tuple[float, str]:
    if bmi >= 35:
        return 18, "very high BMI range"
    if bmi >= 30:
        return 12, "obese BMI range"
    if bmi >= 25:
        return 6, "overweight BMI range"
    return 0, "healthy BMI range"


# Give fasting glucose a point value and explanation label.
def glucose_points(glucose_scale: float, estimated: bool) -> tuple[float, str]:
    prefix = "estimated " if estimated else ""
    if glucose_scale >= 4:
        return 20, f"{prefix}very high fasting glucose range"
    if glucose_scale >= 3:
        return 14, f"{prefix}high fasting glucose range"
    if glucose_scale >= 2:
        return 7, f"{prefix}borderline fasting glucose range"
    return 0, f"{prefix}normal fasting glucose range"


# Convert a probability into a simple risk level.
def reference_level(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    return "High"


# Estimate HDL when the form does not include an entered HDL value.
def default_hdl(sex: float) -> float:
    return 50.0 if sex >= 0.5 else 60.0


# Use entered HDL when available, otherwise estimate it from sex.
def patient_hdl(values: dict[str, float], payload: dict) -> tuple[float, str]:
    entered_hdl = parse_number(payload.get("hdl"))
    if entered_hdl is not None and entered_hdl > 0:
        return float(entered_hdl), "entered HDL"
    return default_hdl(float(values["sex"])), "estimated HDL"


# Calculate the Framingham reference risk for comparison.
def framingham_patient_risk(values: dict[str, float], payload: dict) -> dict:
    # Patient-level Framingham reference calculation kept for internal comparison use.
    age = float(values["age_years"])
    sex_value = float(values["sex"])
    sex = "male" if sex_value >= 0.5 else "female"
    cholesterol = max(float(values["cholesterol"]), 1.0)
    systolic = max(float(values["systolic_bp"]), 1.0)
    hdl, hdl_source = patient_hdl(values, payload)
    coeffs = FRAMINGHAM_LIPIDS[sex]

    if not 30 <= age <= 74:
        eligibility = "Outside usual Framingham age range of 30-74; shown as a rough reference."
    else:
        eligibility = "Within usual Framingham age range. BP treatment status is estimated because this form does not collect it."

    lp = (
        coeffs["age"] * np.log(max(age, 1.0))
        + coeffs["tchol"] * np.log(cholesterol)
        + coeffs["hdl"] * np.log(hdl)
        + coeffs["sbp_untreated"] * np.log(systolic)
        + coeffs["smoking"] * float(values["smoking"])
        + coeffs["diabetes"] * float(values["diabetes"])
    )
    probability = float(1.0 - (coeffs["baseline_survival"] ** np.exp(lp - coeffs["mean_sum"])))
    probability = float(np.clip(probability, 0.0, 0.999))
    return {
        "name": "Framingham",
        "risk_probability": round(probability, 4),
        "risk_percent": round(probability * 100.0, 2),
        "risk_level": reference_level(probability),
        "subtitle": "10-year general CVD lipid score",
        "eligibility_note": eligibility,
        "assumptions": f"This value is already multiplied by 100. Uses {hdl_source} {round(hdl)} mg/dL and assumes untreated systolic BP.",
    }


# Calculate the SCORE2 reference risk for comparison.
def score2_patient_risk(values: dict[str, float], payload: dict) -> dict:
    # Patient-level SCORE2 reference calculation kept for internal comparison use.
    age = float(values["age_years"])
    sex_value = float(values["sex"])
    sex = "male" if sex_value >= 0.5 else "female"
    coeffs = SCORE2_COEFFICIENTS[sex]
    cage = (age - 60.0) / 5.0
    csbp = (float(values["systolic_bp"]) - 120.0) / 20.0
    cholesterol = max(float(values["cholesterol"]), 1.0)
    hdl, hdl_source = patient_hdl(values, payload)
    tchol = cholesterol * MMOL_PER_MGDL_CHOLESTEROL
    hdl_mmol = hdl * MMOL_PER_MGDL_CHOLESTEROL
    ctchol = tchol - 6.0
    chdl = (hdl_mmol - 1.3) / 0.5
    smoking = float(values["smoking"])
    diabetes = float(values["diabetes"])

    if not 40 <= age <= 69:
        eligibility = "Outside usual SCORE2 age range of 40-69; shown as a rough reference."
    else:
        eligibility = "Within usual SCORE2 age range. Low-risk region calibration is used for comparison."

    linear_predictor = (
        coeffs["age"] * cage
        + coeffs["smoking"] * smoking
        + coeffs["sbp"] * csbp
        + coeffs["diabetes"] * diabetes
        + coeffs["tchol"] * ctchol
        + coeffs["hdl"] * chdl
        + coeffs["age_smoking"] * cage * smoking
        + coeffs["age_sbp"] * cage * csbp
        + coeffs["age_tchol"] * cage * ctchol
        + coeffs["age_hdl"] * cage * chdl
        + coeffs["age_diabetes"] * cage * diabetes
    )
    uncalibrated = 1.0 - (coeffs["baseline_survival"] ** np.exp(linear_predictor))
    uncalibrated = min(max(float(uncalibrated), 1e-8), 1.0 - 1e-8)
    probability = float(
        1.0
        - np.exp(
            -np.exp(
                coeffs["scale_1"]["low"]
                + coeffs["scale_2"]["low"] * np.log(-np.log(1.0 - uncalibrated))
            )
        )
    )
    probability = float(np.clip(probability, 0.0, 0.999))
    return {
        "name": "SCORE2",
        "risk_probability": round(probability, 4),
        "risk_percent": round(probability * 100.0, 2),
        "risk_level": reference_level(probability),
        "subtitle": "10-year fatal/non-fatal CVD, low-risk region",
        "eligibility_note": eligibility,
        "assumptions": f"This value is already multiplied by 100. Uses low-risk European region calibration and {hdl_source} {round(hdl)} mg/dL.",
    }


# Build the list of project, Framingham, and SCORE2 scores.
def patient_reference_scores(x_filled: np.ndarray, project_probability: float, payload: dict) -> list[dict]:
    fields = MODEL["question_fields"]
    values = dict(zip(fields, x_filled))
    project = {
        "name": "Project model",
        "risk_probability": round(float(project_probability), 4),
        "risk_percent": round(float(project_probability) * 100.0, 2),
        "risk_level": reference_level(project_probability),
        "subtitle": "Local CAD screening probability, not a 10-year clinical-risk equation",
        "eligibility_note": "Uses the app's available questionnaire inputs, including BMI, diastolic BP, glucose estimate, and family history.",
        "assumptions": "This value is already multiplied by 100. It is trained on the project dataset, so it should be compared as a screening score rather than as the same 10-year outcome used by Framingham/SCORE2.",
    }
    return [project, framingham_patient_risk(values, payload), score2_patient_risk(values, payload)]



#CALCULATE TRANSPARENT CLINICAL SCORE#
def clinical_risk_score(x_filled: np.ndarray, estimated_fields: dict, display_overrides: dict) -> tuple[float, list[dict]]:
    # Transparent points-style score used for explanations and factor cards.
    fields = MODEL["question_fields"]
    values = dict(zip(fields, x_filled))

    age = float(values["age_years"])
    sex = float(values["sex"])
    bmi = float(values["bmi"])
    systolic = float(values["systolic_bp"])
    diastolic = float(values["diastolic_bp"])
    cholesterol = float(values["cholesterol"])
    glucose = float(values["fasting_glucose"])
    smoking = float(values["smoking"])
    diabetes = float(values["diabetes"])
    family_history = float(values["family_history_cvd"])

    bp_score, bp_detail = bp_points(systolic, diastolic)
    chol_score, chol_detail = cholesterol_points(cholesterol)
    bmi_score, bmi_detail = bmi_points(bmi)
    glucose_score, glucose_detail = glucose_points(glucose, FASTING_GLUCOSE_FIELD in estimated_fields)

    components = [
        component("age_years", "Age", round(age, 1), age_points(age), "risk rises with age"),
        component("sex", "Sex", "male" if sex >= 0.5 else "female", 5 if sex >= 0.5 else 0, "male sex is weighted as a CAD risk factor"),
        component("blood_pressure", "Blood pressure", f"{round(systolic)}/{round(diastolic)}", bp_score, bp_detail),
        component("bmi", "BMI", round(bmi, 1), bmi_score, bmi_detail),
        component("cholesterol", "Total cholesterol", round(cholesterol, 1), chol_score, chol_detail),
        component(
            "fasting_glucose",
            "Fasting glucose",
            glucose_display_label(glucose) if FASTING_GLUCOSE_FIELD in estimated_fields else display_overrides.get(FASTING_GLUCOSE_FIELD, round(glucose, 1)),
            glucose_score,
            glucose_detail,
        ),
        component("smoking", "Smoking", "yes" if smoking >= 0.5 else "no", 16 if smoking >= 0.5 else 0, "current smoking is a major modifiable CAD risk factor"),
        component("diabetes", "Diabetes", "yes" if diabetes >= 0.5 else "no", 18 if diabetes >= 0.5 else 0, "diabetes increases cardiovascular risk"),
        component("family_history_cvd", "Family history", "yes" if family_history >= 0.5 else "no", 10 if family_history >= 0.5 else 0, "family history increases screening concern"),
    ]

    total_points = sum(item["contribution"] for item in components)
    probability = float(sigmoid((total_points - 75.0) / 25.0))
    top_components = sorted(components, key=lambda item: item["contribution"], reverse=True)[:5]
    return probability, top_components


# Create one standardized suggestion dictionary.
def suggestion(title: str, detail: str, priority: str = "Helpful", key: str = "general") -> dict:
    # Standard shape for one advice card.
    return {
        "key": key,
        "title": title,
        "detail": detail,
        "priority": priority,
    }


# Choose risk-reduction suggestions based on the patient's risk factors.
def make_reduction_suggestions(x_filled: np.ndarray, missing_fields: list[str], risk_level: str) -> list[dict]:
    # Recommend only actions that match the patient's entered risk factors.
    fields = MODEL["question_fields"]
    values = dict(zip(fields, x_filled))
    missing = set(missing_fields)
    suggestions = []

    systolic = values["systolic_bp"]
    diastolic = values["diastolic_bp"]
    cholesterol = values["cholesterol"]
    glucose = values["fasting_glucose"]
    bmi = values["bmi"]
    smoking = values["smoking"]
    diabetes = values["diabetes"]
    family_history = values["family_history_cvd"]

    if "systolic_bp" not in missing and "diastolic_bp" not in missing and (systolic >= 130 or diastolic >= 80):
        suggestions.append(
            suggestion(
                "Manage blood pressure",
                "Your blood pressure value may be adding to the score. Recheck it regularly and discuss lifestyle or treatment options with a clinician.",
                "High impact",
                "blood_pressure",
            )
        )

    if "cholesterol" not in missing and cholesterol >= 200:
        suggestions.append(
            suggestion(
                "Review cholesterol",
                "The cholesterol value may be increasing risk. A heart-healthier eating pattern and a cholesterol review with a clinician may help.",
                "High impact",
                "cholesterol",
            )
        )

    if "smoking" not in missing and smoking >= 0.5:
        suggestions.append(
            suggestion(
                "Stop smoking or nicotine use",
                "Smoking is a major modifiable CAD risk factor. Quitting support, counseling, or nicotine treatment can make stopping more achievable.",
                "High impact",
                "smoking",
            )
        )

    if "diabetes" not in missing and diabetes >= 0.5:
        suggestions.append(
            suggestion(
                "Control blood sugar",
                "Diabetes can raise CAD risk. Blood sugar monitoring, medication adherence, nutrition, and activity plans should be reviewed with a clinician.",
                "High impact",
                "diabetes",
            )
        )
    elif "fasting_glucose" not in missing and glucose >= 3:
        suggestions.append(
            suggestion(
                "Check fasting glucose",
                "The fasting glucose value is high enough to deserve follow-up testing and advice from a healthcare professional.",
                "Medium impact",
                "glucose",
            )
        )

    if "bmi" not in missing and bmi >= 25:
        suggestions.append(
            suggestion(
                "Work toward a healthier weight",
                "BMI may be contributing to the score. Small, steady changes in food choices and activity can support blood pressure, glucose, and cholesterol.",
                "Medium impact",
                "bmi",
            )
        )

    suggestions.append(
        suggestion(
            "Build weekly activity",
            "Aim for regular movement, such as brisk walking, cycling, or similar moderate activity, if it is safe for you.",
            "Helpful",
            "activity",
        )
    )
    suggestions.append(
        suggestion(
            "Improve daily habits",
            "A heart-healthy diet, enough sleep, and routine checkups can reduce long-term cardiovascular risk factors.",
            "Helpful",
            "lifestyle",
        )
    )

    if "family_history_cvd" not in missing and family_history >= 0.5:
        suggestions.append(
            suggestion(
                "Mention family history",
                "Family history cannot be changed, but it is useful information for screening, earlier monitoring, and prevention planning.",
                "Awareness",
                "family_history",
            )
        )

    if risk_level == "High":
        suggestions.insert(
            0,
            suggestion(
                "Arrange clinical follow-up",
                "Because the result is in the high range, it would be sensible to review the score and risk factors with a healthcare professional.",
                "Priority",
                "clinical_followup",
            ),
        )

    return suggestions[:6]



#MAKE FINAL PREDICTION RESPONSE#
# Run the model and convert its score into a readable risk result.
def make_prediction(payload: dict):
    # Main prediction response used by the browser.
    transformed, missing_fields, estimated_fields, display_overrides, x_filled, x_full = preprocess_payload(payload)
    clinical_probability, top_factors = clinical_risk_score(x_filled, estimated_fields, display_overrides)
    auc_probability = auc_model_probability(payload)
    # Use the transparent clinical-style score as the displayed risk. The
    # AUC-optimized model can overestimate low-risk young users when optional
    # lipid/activity fields are missing and filled with training medians.
    probability = clinical_probability
    percent = float(round(probability * 100.0, 2))

    risk_level = reference_level(probability)

    return {
        "risk_probability": float(round(probability, 4)),
        "risk_percent": percent,
        "risk_level": risk_level,
        "missing_fields": missing_fields,
        "estimated_fields": estimated_fields,
        "model_notes": [
            "The main percentage uses a transparent clinical-style score so low-risk profiles are not overestimated by missing optional model fields.",
            f"The AUC-optimized model estimate for this input is {round(float(auc_probability) * 100.0, 2)}% and is kept as supporting context only."
            if auc_probability is not None
            else "The AUC-optimized model file was not available, so only the transparent score was used.",
            FASTING_GLUCOSE_NOTE,
        ]
        if estimated_fields
        else [
            "The main percentage uses a transparent clinical-style score so low-risk profiles are not overestimated by missing optional model fields.",
            f"The AUC-optimized model estimate for this input is {round(float(auc_probability) * 100.0, 2)}% and is kept as supporting context only."
            if auc_probability is not None
            else "The AUC-optimized model file was not available, so only the transparent score was used.",
        ],
        "why_this_score": risk_reason(top_factors, risk_level),
        "reference_scores": patient_reference_scores(x_filled, probability, payload),
        "top_factors": top_factors,
        "risk_reduction_suggestions": make_reduction_suggestions(x_filled, missing_fields, risk_level),
        "disclaimer": "This tool is for educational screening support only and is not a medical diagnosis.",
    }


class CADRiskHandler(BaseHTTPRequestHandler):
    # Tiny HTTP API and static file server for the local web app.
    # Send a normal HTTP response body.
    def _send_bytes(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Send a JSON response to the browser.
    def _send_json(self, data: dict, status: int = 200):
        self._send_bytes(json.dumps(data).encode("utf-8"), status=status, content_type="application/json; charset=utf-8")

    # Serve files from the webapp folder, such as HTML, CSS, and JavaScript.
    def _serve_static(self, relative_path: str):
        safe_path = relative_path.lstrip("/") or "index.html"
        file_path = (STATIC_DIR / safe_path).resolve()
        if STATIC_DIR.resolve() not in file_path.parents and file_path != STATIC_DIR.resolve():
            self._send_bytes(b"Not found", status=404, content_type="text/plain; charset=utf-8")
            return
        if not file_path.exists() or not file_path.is_file():
            self._send_bytes(b"Not found", status=404, content_type="text/plain; charset=utf-8")
            return

        content_type = "text/plain; charset=utf-8"
        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"

        self._send_bytes(file_path.read_bytes(), content_type=content_type)

    # Handle browser requests for pages and model information.
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"status": "ok", "model": MODEL["model_name"]})
            return
        if parsed.path == "/api/model-info":
            self._send_json(
                {
                    "model_name": MODEL["model_name"],
                    "fields": MODEL["question_fields"],
                    "feature_descriptions": MODEL["feature_descriptions"],
                    "best_hyperparameters": MODEL["best_hyperparameters"],
                }
            )
            return
        if parsed.path == "/api/comparison":
            self._send_json(load_comparison_summary())
            return
        if parsed.path == "/":
            self._serve_static("index.html")
            return
        self._serve_static(parsed.path)

    # Handle prediction requests sent from the web form.
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self._send_json({"error": "Endpoint not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        validation_errors = validate_payload(payload)
        if validation_errors:
            self._send_json({"error": "Invalid input range", "messages": validation_errors}, status=400)
            return

        result = make_prediction(payload)
        self._send_json(result)


# Start the local web server.
def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}. Run python models/train_cad_risk_model.py first.")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), CADRiskHandler)
    print("CAD risk app running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
