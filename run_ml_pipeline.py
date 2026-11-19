
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    "src.preprocessing.data_cleaner",
    "src.preprocessing.feature_engineering",
    "src.models.baseline_model",
]


def main():
    print("=" * 60)
    print("NYC 311 MACHINE LEARNING PIPELINE")
    print("=" * 60)

    for step in STEPS:
        print(f"\n\nRunning: {step}")
        print("-" * 60)

        subprocess.run(
            [sys.executable, "-m", step],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print("\n" + "=" * 60)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
