
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "nyc311_model_data.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
EVALUATION_DIR = PROJECT_ROOT / "models" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "delay_flag"

CATEGORICAL_CANDIDATES = [
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "borough",
    "city",
]

NUMERIC_CANDIDATES = [
    "latitude",
    "longitude",
    "created_year",
    "created_month",
    "created_day_of_week",
    "created_hour",
    "is_weekend",
]


def evaluate_predictions(name, y_true, y_pred, y_prob=None):
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
    }

    if y_prob is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def main():
    print("=" * 60)
    print("NYC 311 BASELINE MODEL")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Separate predictors and target
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    categorical_features = [
        column for column in CATEGORICAL_CANDIDATES
        if column in X.columns
    ]

    numeric_features = [
        column for column in NUMERIC_CANDIDATES
        if column in X.columns
    ]

    # Remove columns not explicitly permitted as predictors
    X = X[categorical_features + numeric_features]

    print("Dataset shape:", df.shape)
    print("Categorical features:", categorical_features)
    print("Numeric features:", numeric_features)
    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_features),
        ("numeric", numeric_pipeline, numeric_features),
    ])

    # Majority-class baseline
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)

    dummy_predictions = dummy_model.predict(X_test)

    dummy_metrics = evaluate_predictions(
        "DummyClassifier",
        y_test,
        dummy_predictions,
    )

    # Logistic regression baseline
    logistic_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    logistic_pipeline.fit(X_train, y_train)

    predictions = logistic_pipeline.predict(X_test)
    probabilities = logistic_pipeline.predict_proba(X_test)[:, 1]

    logistic_metrics = evaluate_predictions(
        "LogisticRegression",
        y_test,
        predictions,
        probabilities,
    )

    print("\n" + "-" * 60)
    print("DUMMY CLASSIFIER RESULTS")
    print("-" * 60)
    print(dummy_metrics["classification_report"])
    print("Accuracy:", round(dummy_metrics["accuracy"], 4))
    print("F1:", round(dummy_metrics["f1_score"], 4))

    print("\n" + "-" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("-" * 60)
    print(logistic_metrics["classification_report"])
    print("Accuracy:", round(logistic_metrics["accuracy"], 4))
    print("F1:", round(logistic_metrics["f1_score"], 4))
    if "roc_auc" in logistic_metrics:
        print("ROC-AUC:", round(logistic_metrics["roc_auc"], 4))

    print("\nConfusion Matrix:")
    print(logistic_metrics["confusion_matrix"])

    # Save trained pipeline
    model_file = MODEL_DIR / "nyc311_logistic_baseline.joblib"
    joblib.dump(logistic_pipeline, model_file)

    # Save evaluation results
    results = {
        "dummy_classifier": dummy_metrics,
        "logistic_regression": logistic_metrics,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": {
            "categorical": categorical_features,
            "numeric": numeric_features,
        },
        "random_state": 42,
        "test_size": 0.2,
    }

    results_file = EVALUATION_DIR / "baseline_metrics.json"
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nModel saved to:", model_file)
    print("Metrics saved to:", results_file)
    print("\nBASELINE TRAINING COMPLETE")


if __name__ == "__main__":
    main()
