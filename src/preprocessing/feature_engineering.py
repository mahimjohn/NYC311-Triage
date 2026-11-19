
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_cleaned.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_model_data.csv"

# Features available when the 311 request is created
BASE_FEATURES = [
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "borough",
    "city",
    "latitude",
    "longitude",
]

TARGET_COLUMN = "delay_flag"


def engineer_features():
    print("=" * 60)
    print("NYC 311 FEATURE ENGINEERING")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )

    # Time-based features available at request creation
    df["created_year"] = df["created_date"].dt.year
    df["created_month"] = df["created_date"].dt.month
    df["created_day_of_week"] = df["created_date"].dt.dayofweek
    df["created_hour"] = df["created_date"].dt.hour
    df["is_weekend"] = (
        df["created_day_of_week"] >= 5
    ).astype(int)

    # Keep only available predictors
    selected_features = [
        column for column in BASE_FEATURES
        if column in df.columns
    ]

    time_features = [
        "created_year",
        "created_month",
        "created_day_of_week",
        "created_hour",
        "is_weekend",
    ]

    final_columns = (
        selected_features
        + time_features
        + [TARGET_COLUMN]
    )

    model_df = df[final_columns].copy()

    # Save model-ready table
    model_df.to_csv(OUTPUT_FILE, index=False)

    print("Selected categorical/numeric features:")
    print(selected_features + time_features)

    print("\nTarget distribution:")
    print(model_df[TARGET_COLUMN].value_counts())

    print("\nFinal shape:", model_df.shape)
    print("Saved to:", OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    engineer_features()
