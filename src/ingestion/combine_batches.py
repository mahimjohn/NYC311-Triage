from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BATCH_DIR = PROJECT_ROOT / "data" / "raw" / "paginated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "nyc311_master_100k.csv"


def combine_batches():
    """Combine downloaded CSV batches into one master dataset."""

    batch_files = sorted(BATCH_DIR.glob("nyc311_batch_*.csv"))

    if not batch_files:
        raise FileNotFoundError(
            f"No CSV batch files found in {BATCH_DIR}"
        )

    print(f"Batch files found: {len(batch_files)}")

    dataframes = []

    for file in batch_files:
        print(f"Reading: {file.name}")

        df = pd.read_csv(file)
        dataframes.append(df)

    # Combine all batches
    master_df = pd.concat(dataframes, ignore_index=True)

    print("\n========== MASTER DATASET ==========")
    print("Total rows:", len(master_df))
    print("Total columns:", len(master_df.columns))

    # Check duplicate IDs
    duplicate_ids = master_df["unique_key"].duplicated().sum()

    print("Duplicate unique_keys:", duplicate_ids)

    # Check date range
    dates = pd.to_datetime(
        master_df["created_date"],
        errors="coerce"
    )

    print("Earliest record:", dates.min())
    print("Latest record:", dates.max())

    # Save master dataset
    master_df.to_csv(OUTPUT_FILE, index=False)

    print("\nMaster dataset saved successfully!")
    print("Output:", OUTPUT_FILE)
    print("File size:", f"{OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")
    print("\n========== COMPLETE ==========")


if __name__ == "__main__":
    combine_batches()