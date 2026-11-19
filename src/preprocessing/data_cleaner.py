
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stratified"
    / "nyc311_stratified_sample.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "nyc311_cleaned.csv"

TARGET_THRESHOLD_DAYS = 7


def clean_data():
    print("=" * 60)
    print("NYC 311 DATA CLEANING")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    initial_rows = len(df)

    print("Initial rows:", initial_rows)

    # Remove duplicate request IDs
    if "unique_key" in df.columns:
        df = df.drop_duplicates(subset="unique_key")

    # Parse dates
    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )
    df["closed_date"] = pd.to_datetime(
        df["closed_date"], errors="coerce"
    )

    # Remove rows without valid creation dates
    df = df.dropna(subset=["created_date"])

    # Only closed records can be used for supervised learning labels
    df = df.dropna(subset=["closed_date"])

    # Calculate resolution duration in days
    duration = (
        df["closed_date"] - df["created_date"]
    ).dt.total_seconds() / 86400

    df["resolution_days"] = duration

    # Remove invalid negative durations
    df = df[df["resolution_days"] >= 0].copy()

    # Create binary target: 1 if resolved in more than 7 days
    df["delay_flag"] = (
        df["resolution_days"] > TARGET_THRESHOLD_DAYS
    ).astype(int)

    # Normalize text fields: trim surrounding whitespace
    text_columns = df.select_dtypes(include="object").columns

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # Save cleaned data
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCLEANING SUMMARY")
    print("-" * 60)
    print("Initial records:", initial_rows)
    print("Final records:", len(df))
    print("Removed records:", initial_rows - len(df))
    print("Valid resolution duration:", df["resolution_days"].notna().sum())

    print("\nTarget distribution:")
    print(df["delay_flag"].value_counts())

    print("\nTarget percentage:")
    print(
        (df["delay_flag"].value_counts(normalize=True) * 100).round(2)
    )

    print("\nSaved to:", OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    clean_data()
