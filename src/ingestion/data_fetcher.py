import os
import json
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

from src.config.config_loader import load_config, get_config_value

load_dotenv()

CONFIG = load_config()

API_URL = get_config_value(
    CONFIG,
    "data_sources.api_endpoints.json"
)

APP_TOKEN = os.getenv("NYC_OPEN_DATA_APP_TOKEN")

START_DATE = get_config_value(CONFIG, "ingestion.start_date")
END_DATE = get_config_value(CONFIG, "ingestion.end_date")

PAGE_SIZE = get_config_value(CONFIG, "ingestion.page_size")
MAX_ROWS = get_config_value(CONFIG, "ingestion.max_rows")

REQUEST_TIMEOUT = get_config_value(CONFIG, "ingestion.request_timeout")
RETRY_COUNT = get_config_value(CONFIG, "ingestion.retry_count")
RETRY_DELAY = get_config_value(CONFIG, "ingestion.retry_delay")
EXPONENTIAL_BACKOFF = get_config_value(
    CONFIG,
    "ingestion.exponential_backoff"
)

RAW_DATA_DIR = Path(
    get_config_value(CONFIG, "paths.data.raw")
)

OUTPUT_DIR = RAW_DATA_DIR / "paginated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_page(params, headers):
    """Request one page from NYC Open Data with retries."""

    max_attempts = RETRY_COUNT + 1

    for attempt in range(max_attempts):
        try:
            response = requests.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as error:

            if attempt == max_attempts - 1:
                raise

            if EXPONENTIAL_BACKOFF:
                wait_time = RETRY_DELAY * (2 ** attempt)
            else:
                wait_time = RETRY_DELAY

            print(
                f"Request failed: {error}\n"
                f"Retrying in {wait_time} seconds "
                f"({attempt + 1}/{RETRY_COUNT})..."
            )

            time.sleep(wait_time)

def fetch_all_batches():

    headers = {
        "Accept": "application/json"
    }

    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    where_clause = (
        f"created_date >= '{START_DATE}' "
        f"AND created_date < '{END_DATE}'"
    )

    offset = 0
    total_downloaded = 0
    batch_number = 1

    print("=" * 60)
    print("NYC 311 PAGINATED DATA INGESTION")
    print("=" * 60)
    print(f"Start date: {START_DATE}")
    print(f"End date:   {END_DATE}")
    print(f"Page size:  {PAGE_SIZE:,}")
    print(f"Max rows:   {MAX_ROWS:,}")
    print("=" * 60)

    while total_downloaded < MAX_ROWS:

        remaining_rows = MAX_ROWS - total_downloaded

        current_limit = min(
            PAGE_SIZE,
            remaining_rows
        )

        params = {
            "$where": where_clause,
            "$limit": current_limit,
            "$offset": offset,
            "$order": "created_date ASC, unique_key ASC"
        }

        print(f"\nRequesting batch {batch_number:03d}")
        print(f"Offset: {offset:,}")
        print(f"Limit:  {current_limit:,}")

        records = fetch_page(params, headers)

        if not records:
            print("\nAPI returned no more records.")
            break

        df = pd.DataFrame(records)

        # Save this batch independently.
        batch_name = f"nyc311_batch_{batch_number:03d}"

        json_path = OUTPUT_DIR / f"{batch_name}.json"
        csv_path = OUTPUT_DIR / f"{batch_name}.csv"

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=2
            )

        df.to_csv(csv_path, index=False)

        records_count = len(records)

        total_downloaded += records_count

        print(f"Records received: {records_count:,}")
        print(f"JSON saved: {json_path}")
        print(f"CSV saved:  {csv_path}")
        print(f"Total downloaded so far: {total_downloaded:,}")

        offset += records_count
        batch_number += 1

        if records_count < current_limit:
            print("\nReceived a partial batch. Download appears complete.")
            break

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total records downloaded: {total_downloaded:,}")
    print(f"Batches downloaded:      {batch_number - 1}")
    print(f"Output directory:        {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    fetch_all_batches()