from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_master_100k.csv"


def profile_dataset():
    """Profile the master NYC 311 dataset."""

    df = pd.read_csv(DATA_FILE, low_memory=False)

    print("=" * 60)
    print("NYC 311 DATA PROFILING")
    print("=" * 60)

    # Basic information
    print("\n--- BASIC INFORMATION ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Memory usage (MB):", round(df.memory_usage(deep=True).sum() / 1024**2, 2))

    # Column information
    print("\n--- COLUMN INFORMATION ---")
    print(df.dtypes.to_string())

    # Missing values
    print("\n--- MISSING VALUES ---")
    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2)
    })

    missing = missing.sort_values(
        "missing_percent",
        ascending=False
    )

    print(missing.to_string())

    # Unique values
    print("\n--- UNIQUE VALUES ---")
    print(df.nunique().sort_values().to_string())

    # Duplicate rows
    print("\n--- DUPLICATE CHECK ---")
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate unique_keys:", df["unique_key"].duplicated().sum())

    # Date profiling
    print("\n--- DATE PROFILING ---")

    for column in ["created_date", "closed_date"]:
        if column in df.columns:
            dates = pd.to_datetime(df[column], errors="coerce")
            print(f"\n{column}")
            print("Invalid dates:", dates.isna().sum())
            print("Earliest:", dates.min())
            print("Latest:", dates.max())

    # Outcome availability
    print("\n--- RESOLUTION INFORMATION ---")
    print("Records with closed_date:", df["closed_date"].notna().sum())
    print("Records without closed_date:", df["closed_date"].isna().sum())

    # Categorical distributions
    for column in ["agency", "complaint_type", "borough"]:
        if column in df.columns:
            print(f"\n--- TOP VALUES: {column} ---")
            print(df[column].value_counts(dropna=False).head(15).to_string())

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    profile_dataset()