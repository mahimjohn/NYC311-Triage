
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "nyc311_logistic_baseline.joblib"
)

THRESHOLD = 0.5


def prepare_request(request_data):
    """Convert a new request into model-compatible features."""

    created_date = pd.to_datetime(
        request_data.get("created_date"),
        errors="coerce"
    )

    if pd.isna(created_date):
        raise ValueError("A valid created_date is required.")

    features = {
        "agency": request_data.get("agency"),
        "complaint_type": request_data.get("complaint_type"),
        "descriptor": request_data.get("descriptor"),
        "location_type": request_data.get("location_type"),
        "incident_zip": request_data.get("incident_zip"),
        "borough": request_data.get("borough"),
        "city": request_data.get("city"),
        "latitude": request_data.get("latitude"),
        "longitude": request_data.get("longitude"),
        "created_year": created_date.year,
        "created_month": created_date.month,
        "created_day_of_week": created_date.dayofweek,
        "created_hour": created_date.hour,
        "is_weekend": int(created_date.dayofweek >= 5),
    }

    return pd.DataFrame([features])


def predict_delay(request_data):
    """Predict whether a request may take more than 7 days."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. "
            "Train the baseline model first."
        )

    model = joblib.load(MODEL_FILE)

    features = prepare_request(request_data)

    probability = model.predict_proba(features)[0][1]
    prediction = int(probability >= THRESHOLD)

    return {
        "delay_probability": round(float(probability), 4),
        "delay_prediction": prediction,
        "prediction_label": (
            "Delayed (>7 days)"
            if prediction == 1
            else "Not delayed (<=7 days)"
        ),
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    # Example request — illustrative values only
    example_request = {
        "created_date": "2026-11-19T10:30:00",
        "agency": "NYPD",
        "complaint_type": "Noise - Residential",
        "descriptor": "Loud Music/Party",
        "location_type": "Residential Building",
        "incident_zip": "10001",
        "borough": "MANHATTAN",
        "city": "NEW YORK",
        "latitude": 40.7500,
        "longitude": -73.9900,
    }

    result = predict_delay(example_request)

    print("=" * 50)
    print("NYC 311 DELAY PREDICTION")
    print("=" * 50)
    print(result)
