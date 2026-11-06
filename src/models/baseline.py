
import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.config.config_loader import get_project_root
from src.features.feature_engineering import (
    prepare_training_data,
    CATEGORICAL_FEATURES
)


TARGET_COLUMN = "delay_flag"
GROUP_COLUMN = "agency"
DEFAULT_K = 100


def calculate_precision_at_k(y_true, scores, k=DEFAULT_K):
    """Calculate Precision@K using descending risk scores."""

    if k <= 0:
        raise ValueError("K must be greater than zero.")

    if len(y_true) != len(scores):
        raise ValueError(
            "Actual labels and scores must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation dataset is empty.")

    k = min(k, len(y_true))

    evaluation = pd.DataFrame({
        "actual": list(y_true),
        "score": list(scores)
    })

    top_k = evaluation.sort_values(
        "score",
        ascending=False,
        kind="mergesort"
    ).head(k)

    return float(top_k["actual"].sum() / k)


def calculate_classification_metrics(y_true, predictions):
    """Calculate standard classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).tolist()
    }


def evaluate_baseline(
    name,
    y_true,
    predictions,
    ranking_scores,
    k=DEFAULT_K
):
    """Evaluate a baseline using classification metrics and Precision@K."""

    metrics = calculate_classification_metrics(
        y_true,
        predictions
    )

    metrics["precision_at_k"] = calculate_precision_at_k(
        y_true,
        ranking_scores,
        k
    )

    metrics["k"] = min(k, len(y_true))
    metrics["model"] = name

    return metrics


def build_level_0_baseline(y_train, y_test):
    """Predict the global majority class for every request."""

    majority_class = int(
        y_train.value_counts().idxmax()
    )

    global_delayed_rate = float(y_train.mean())

    predictions = [
        majority_class
    ] * len(y_test)

    ranking_scores = [
        global_delayed_rate
    ] * len(y_test)

    return predictions, ranking_scores


def build_level_1_baseline(
    X_train,
    y_train,
    X_test
):
    """Predict using the majority delayed class within each agency."""

    if GROUP_COLUMN not in X_train.columns:
        raise ValueError(
            f"Required grouping column missing: {GROUP_COLUMN}"
        )

    if GROUP_COLUMN not in X_test.columns:
        raise ValueError(
            f"Required grouping column missing: {GROUP_COLUMN}"
        )

    training_data = pd.DataFrame({
        GROUP_COLUMN: X_train[GROUP_COLUMN].fillna("Unknown"),
        TARGET_COLUMN: y_train
    })

    # Calculate agency-level statistics using training data only.
    agency_statistics = training_data.groupby(
        GROUP_COLUMN
    )[TARGET_COLUMN].agg(
        delayed_rate="mean",
        majority_class=lambda values: int(
            values.value_counts().idxmax()
        )
    )

    global_delayed_rate = float(y_train.mean())
    global_majority_class = int(
        y_train.value_counts().idxmax()
    )

    test_agencies = X_test[GROUP_COLUMN].fillna("Unknown")

    predictions = test_agencies.map(
        agency_statistics["majority_class"]
    ).fillna(global_majority_class).astype(int)

    ranking_scores = test_agencies.map(
        agency_statistics["delayed_rate"]
    ).fillna(global_delayed_rate)

    return (
        predictions.tolist(),
        ranking_scores.tolist(),
        agency_statistics
    )


def save_baseline_results(results):
    """Save baseline evaluation results as JSON."""

    project_root = get_project_root()

    output_directory = project_root / "models" / "evaluation"
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_directory / "baseline_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    return output_path


def run_baseline_evaluation():
    """Execute Level 0 and Level 1 baseline evaluation."""

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _
    ) = prepare_training_data()

    print("\nNYC 311 BASELINE EVALUATION")
    print("=" * 50)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Level 0
    level_0_predictions, level_0_scores = (
        build_level_0_baseline(y_train, y_test)
    )

    level_0_results = evaluate_baseline(
        "Level 0 - Global Majority",
        y_test,
        level_0_predictions,
        level_0_scores
    )

    # Level 1
    (
        level_1_predictions,
        level_1_scores,
        agency_statistics
    ) = build_level_1_baseline(
        X_train,
        y_train,
        X_test
    )

    level_1_results = evaluate_baseline(
        "Level 1 - Agency Majority",
        y_test,
        level_1_predictions,
        level_1_scores
    )

    results = {
        "dataset": {
            "training_records": len(X_train),
            "testing_records": len(X_test)
        },
        "level_0": level_0_results,
        "level_1": level_1_results
    }

    print("\nLEVEL 0 RESULTS")
    print("-" * 40)

    for metric, value in level_0_results.items():
        print(f"{metric}: {value}")

    print("\nLEVEL 1 RESULTS")
    print("-" * 40)

    for metric, value in level_1_results.items():
        print(f"{metric}: {value}")

    print("\nAGENCY STATISTICS")
    print("-" * 40)
    print(agency_statistics.head(10))

    output_path = save_baseline_results(results)

    print("\nBaseline results saved:")
    print(output_path)


if __name__ == "__main__":
    run_baseline_evaluation()
