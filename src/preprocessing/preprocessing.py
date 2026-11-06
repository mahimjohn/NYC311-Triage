
import pandas as pd
from pathlib import Path

from src.config.config_loader import load_config, get_project_root


CATEGORICAL_COLUMNS = [
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "location_type",
    "borough",
    "city"
]


def load_raw_data(config):
    """Load the raw NYC 311 dataset."""

    project_root = get_project_root()

    raw_directory = project_root / config["paths"]["data"]["raw"]
    raw_path = raw_directory / "nyc311_2025_sample.csv"

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}"
        )

    return pd.read_csv(raw_path)


def validate_schema(df, config):
    """Validate that all required columns exist."""

    required_columns = config["data"]["required_columns"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )


def preprocess_data(df, config):
    """Clean data and generate the delayed-request target."""

    df = df.copy()

    validate_schema(df, config)

    initial_rows = len(df)

    # Convert date columns
    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )

    df["closed_date"] = pd.to_datetime(
        df["closed_date"], errors="coerce"
    )

    # Remove records with missing identifiers or invalid dates
    df = df.dropna(
        subset=["unique_key", "created_date", "closed_date"]
    ).copy()

    # Handle duplicate request identifiers
    duplicate_count = df["unique_key"].duplicated().sum()

    df = df.drop_duplicates(
        subset=["unique_key"], keep="first"
    ).copy()

    # Fill missing categorical values
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("Unknown")

    # Calculate resolution duration
    df["resolution_days"] = (
        df["closed_date"] - df["created_date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Remove records with invalid chronology
    invalid_duration_count = (
        df["resolution_days"] < 0
    ).sum()

    df = df[
        df["resolution_days"] >= 0
    ].copy()

    # Generate binary target
    threshold = config["target"]["threshold_days"]

    df["delay_flag"] = (
        df["resolution_days"] > threshold
    ).astype(int)

    # Generate processing report
    report = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "removed_rows": initial_rows - len(df),
        "duplicate_keys_found": int(duplicate_count),
        "invalid_durations_removed": int(
            invalid_duration_count
        ),
        "delayed_requests": int(df["delay_flag"].sum()),
        "non_delayed_requests": int(
            (df["delay_flag"] == 0).sum()
        ),
        "delayed_percentage": round(
            df["delay_flag"].mean() * 100, 2
        )
    }

    return df, report


def save_processed_data(df, config):
    """Save the processed dataset."""

    project_root = get_project_root()

    processed_directory = (
        project_root / config["paths"]["data"]["processed"]
    )

    processed_directory.mkdir(
        parents=True, exist_ok=True
    )

    output_path = (
        processed_directory / "nyc311_2025_cleaned.csv"
    )

    df.to_csv(output_path, index=False)

    return output_path


def run_preprocessing():
    """Execute the complete preprocessing pipeline."""

    config = load_config()

    print("\nNYC 311 PREPROCESSING PIPELINE")
    print("=" * 40)

    raw_df = load_raw_data(config)

    print(f"Raw records loaded: {len(raw_df)}")

    processed_df, report = preprocess_data(
        raw_df, config
    )

    output_path = save_processed_data(
        processed_df, config
    )

    print("\nPREPROCESSING REPORT")
    print("=" * 40)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nProcessed dataset saved:")
    print(output_path)

    return processed_df, report


if __name__ == "__main__":
    run_preprocessing()
