from pathlib import Path
import json
import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_cleaned.csv"
MODEL_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_model_data.csv"
EVALUATION_DIR = PROJECT_ROOT / "models" / "evaluation"

TARGET = "delay_flag"

CATEGORICAL_FEATURES = [
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "borough",
    "city",
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "created_year",
    "created_month",
    "created_day_of_week",
    "created_hour",
    "is_weekend",
]


def build_preprocessor():
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ])


def main():
    print("=" * 70)
    print("NYC 311 - TEMPORAL MODEL COMPARISON")
    print("=" * 70)

    cleaned = pd.read_csv(CLEANED_FILE, low_memory=False)
    model_df = pd.read_csv(MODEL_DATA_FILE, low_memory=False)

    if len(cleaned) != len(model_df):
        raise ValueError("Dataset row counts do not match.")

    if not cleaned[TARGET].equals(model_df[TARGET]):
        raise ValueError("Target alignment verification failed.")

    model_df["created_date"] = pd.to_datetime(
        cleaned["created_date"], errors="coerce"
    )

    cutoff = model_df["created_date"].quantile(0.80)

    train_df = model_df[model_df["created_date"] <= cutoff]
    test_df = model_df[model_df["created_date"] > cutoff]

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Temporal cutoff: {cutoff}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, classifier in models.items():
        print(f"\n{'-' * 70}")
        print(f"Training: {name}")
        print("-" * 70)

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_seconds = time.time() - start_time

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "training_seconds": round(training_seconds, 2),
        }

        results[name] = metrics

        for metric, value in metrics.items():
            if metric == "training_seconds":
                print(f"{metric}: {value:.2f} sec")
            else:
                print(f"{metric}: {value:.4f}")

    # Summary table
    results_df = pd.DataFrame(results).T

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(results_df.round(4).to_string())

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    output_file = EVALUATION_DIR / "model_comparison_metrics.json"

    metadata = {
        "experiment": "temporal_model_comparison",
        "cutoff_date": str(cutoff),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_start": str(test_df["created_date"].min()),
        "test_end": str(test_df["created_date"].max()),
        "features": feature_columns,
        "models": results,
    }

    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nResults saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
