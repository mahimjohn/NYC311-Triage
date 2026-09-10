from __future__ import annotations

import yaml

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"

OUTPUT_FILE = RAW_DIR / "nyc311_2025_sample.csv"
METADATA_FILE = SNAPSHOT_DIR / "nyc311_2025_sample_metadata.json"

PAGE_SIZE = 50_000
MAX_ROWS = 100_000

START_DATE = "2025-01-01T00:00:00"
END_DATE = "2026-01-01T00:00:00"


# ---------------------------------------------------------
# Fields available at request creation time
# ---------------------------------------------------------

PREDICTION_TIME_FIELDS = [
    "unique_key",
    "created_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "borough",
    "city",
    "latitude",
    "longitude",
]

# Used only to construct our target later.
TARGET_FIELD = "closed_date"


# ---------------------------------------------------------
# Query construction
# ---------------------------------------------------------

SELECT_FIELDS = ",".join(
    PREDICTION_TIME_FIELDS + [TARGET_FIELD]
)

WHERE_CLAUSE = (
    f"created_date >= '{START_DATE}' "
    f"AND created_date < '{END_DATE}' "
    f"AND closed_date IS NOT NULL"
)


# ---------------------------------------------------------
# Download function
# ---------------------------------------------------------

def fetch_page(
    client: httpx.Client,
    offset: int,
    limit: int,
) -> list[dict]:
    """Fetch one page of NYC 311 records."""

    params = {
        "$select": SELECT_FIELDS,
        "$where": WHERE_CLAUSE,
        "$order": "created_date ASC, unique_key ASC",
        "$limit": limit,
        "$offset": offset,
    }

    response = client.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()

    return response.json()


def download_dataset() -> pd.DataFrame:
    """Download a controlled NYC 311 development extract."""

    records: list[dict] = []
    offset = 0

    with httpx.Client(
        headers={
            "User-Agent": "NYC311-Triage-Capstone/0.1"
        }
    ) as client:

        while len(records) < MAX_ROWS:

            remaining = MAX_ROWS - len(records)
            limit = min(PAGE_SIZE, remaining)

            print(
                f"Requesting rows {offset:,} "
                f"to {offset + limit - 1:,}..."
            )

            page = fetch_page(
                client=client,
                offset=offset,
                limit=limit,
            )

            if not page:
                print("API returned no more records.")
                break

            records.extend(page)

            print(
                f"Received {len(page):,} rows. "
                f"Total: {len(records):,}"
            )

            if len(page) < limit:
                print("Reached end of available records.")
                break

            offset += limit

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic ingestion validation checks."""

    required_columns = set(
        PREDICTION_TIME_FIELDS + [TARGET_FIELD]
    )

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    duplicate_keys = df["unique_key"].duplicated().sum()

    if duplicate_keys:
        raise ValueError(
            f"Found {duplicate_keys:,} duplicate unique_key values."
        )

    print("\nValidation passed.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Duplicate Unique Keys: "
        f"{duplicate_keys:,}"
    )


# ---------------------------------------------------------
# Save snapshot
# ---------------------------------------------------------

def save_snapshot(df: pd.DataFrame) -> None:
    """Save data and reproducibility metadata."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    metadata = {
        "dataset": "311 Service Requests from 2020 to Present",
        "dataset_id": "erm2-nwe9",
        "source_url": (
            "https://data.cityofnewyork.us/"
            "Social-Services/"
            "311-Service-Requests-from-2020-to-Present/"
            "erm2-nwe9"
        ),
        "api_url": BASE_URL,
        "extraction_timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "requested_max_rows": MAX_ROWS,
        "actual_rows": len(df),
        "actual_columns": len(df.columns),
        "columns": list(df.columns),
        "target_field": TARGET_FIELD,
        "prediction_time_fields": PREDICTION_TIME_FIELDS,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nSnapshot saved:")
    print(OUTPUT_FILE)

    print("\nMetadata saved:")
    print(METADATA_FILE)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("NYC 311 DATA INGESTION")
    print("=" * 60)

    df = download_dataset()

    validate_dataset(df)

    save_snapshot(df)

    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()