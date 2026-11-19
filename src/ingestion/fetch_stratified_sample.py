
from pathlib import Path
from datetime import datetime, timedelta
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config.config_loader import load_config, get_config_value

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config()

API_URL = get_config_value(
    CONFIG, "data_sources.api_endpoints.json"
)

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 11, 20)  # exclusive

WINDOW_DAYS = 7
LIMIT_PER_WINDOW = 300

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "stratified"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

APP_TOKEN = os.getenv("NYC_OPEN_DATA_APP_TOKEN")

HEADERS = {}
if APP_TOKEN:
    HEADERS["X-App-Token"] = APP_TOKEN


def fetch_window(start, end):
    """Fetch a limited sample of records for one time window."""

    where_clause = (
        f"created_date >= '{start.strftime('%Y-%m-%dT%H:%M:%S')}' "
        f"AND created_date < '{end.strftime('%Y-%m-%dT%H:%M:%S')}'"
    )

    params = {
        "$where": where_clause,
        "$limit": LIMIT_PER_WINDOW,
        "$order": "created_date DESC, unique_key DESC"
    }

    for attempt in range(4):
        try:
            response = requests.get(
                API_URL,
                params=params,
                headers=HEADERS,
                timeout=30
            )

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            wait_time = 2 ** attempt
            print(f"Request failed: {error}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print(f"FAILED window: {start} to {end}")
    return []


def main():
    all_records = []
    current = START_DATE
    window_number = 0
    failed_windows = []

    print("=" * 60)
    print("NYC 311 TIME-STRATIFIED DATA COLLECTION")
    print("=" * 60)

    while current < END_DATE:
        window_end = min(
            current + timedelta(days=WINDOW_DAYS),
            END_DATE
        )

        window_number += 1

        print(
            f"\nWindow {window_number}: "
            f"{current.date()} -> {window_end.date()}"
        )

        records = fetch_window(current, window_end)

        if records:
            all_records.extend(records)
            print(f"Retrieved: {len(records)} records")
        else:
            failed_windows.append(
                (current.date(), window_end.date())
            )
            print("No records retrieved for this window.")

        current = window_end

        # Be considerate of the public API.
        time.sleep(0.15)

    if not all_records:
        print("No records were collected. Exiting.")
        return

    df = pd.DataFrame(all_records)

    # Remove duplicate records across time windows.
    if "unique_key" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="unique_key")
        print(f"\nRemoved duplicates: {before - len(df)}")

    # Sort chronologically for easier inspection.
    if "created_date" in df.columns:
        df["created_date"] = pd.to_datetime(
            df["created_date"], errors="coerce"
        )
        df = df.sort_values("created_date")

    output_file = OUTPUT_DIR / "nyc311_stratified_sample.csv"
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print("Total records:", len(df))
    print("Date range:")
    print(df["created_date"].min())
    print(df["created_date"].max())
    print("Failed windows:", len(failed_windows))
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()
