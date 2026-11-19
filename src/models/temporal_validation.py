from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_cleaned.csv"
MODEL_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_model_data.csv"

ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"
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


def main():
    print("=" * 65)
    print("NYC 311 - TEMPORAL VALIDATION")
    print("=" * 65)

    # Load both datasets
    cleaned = pd.read_csv(CLEANED_FILE, low_memory=False)
    model_df = pd.read_csv(MODEL_DATA_FILE, low_memory=False)

    # Verify row-level alignment before attaching timestamps
    if len(cleaned) != len(model_df):
        raise ValueError("Dataset row counts do not match.")

    if not cleaned[TARGET].equals(model_df[TARGET]):
        raise ValueError("Target columns are not aligned.")

    # Attach creation timestamp to model data
    model_df["created_date"] = pd.to_datetime(
        cleaned["created_date"], errors="coerce"
    )

    if model_df["created_date"].isna().any():
        raise ValueError("Missing timestamps found.")

    # Chronological 80/20 split
    cutoff = model_df["created_date"].quantile(0.80)

    train_df = model_df[model_df["created_date"] <= cutoff].copy()
    test_df = model_df[model_df["created_date"] > cutoff].copy()

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    print(f"\nCutoff date: {cutoff}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(
        f"Training period: {train_df['created_date'].min()} "
        f"to {train_df['created_date'].max()}"
    )
    print(
        f"Testing period: {test_df['created_date'].min()} "
        f"to {test_df['created_date'].max()}"
    )

    # Preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ])

    # Same baseline Logistic Regression configuration
    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # Train only on earlier records
    model.fit(X_train, y_train)

    # Evaluate on later records
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

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
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
        "dataset_rows": len(model_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "cutoff_date": str(cutoff),
        "train_start": str(train_df["created_date"].min()),
        "train_end": str(train_df["created_date"].max()),
        "test_start": str(test_df["created_date"].min()),
        "test_end": str(test_df["created_date"].max()),
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "split_strategy": "chronological_80_20",
    }

    # Print results
    print("\n" + "=" * 65)
    print("TEMPORAL VALIDATION RESULTS")
    print("=" * 65)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        print(f"{metric}: {metrics[metric]:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Save evaluation results separately from baseline
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    output_file = EVALUATION_DIR / "temporal_validation_metrics.json"

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
