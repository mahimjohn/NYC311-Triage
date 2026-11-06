
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.config_loader import load_config, get_project_root


CATEGORICAL_FEATURES = [
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "borough",
    "city"
]

NUMERICAL_FEATURES = [
    "latitude",
    "longitude",
    "created_hour",
    "created_day_of_week",
    "created_month"
]

IDENTIFIER_COLUMNS = [
    "unique_key"
]

LEAKAGE_COLUMNS = [
    "closed_date",
    "resolution_days"
]

TARGET_COLUMN = "delay_flag"


def load_processed_data(config):
    """Load the processed dataset."""

    project_root = get_project_root()

    data_path = (
        project_root
        / config["paths"]["data"]["processed"]
        / "nyc311_2025_cleaned.csv"
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}"
        )

    return pd.read_csv(data_path)


def validate_features(df):
    """Validate required feature and target columns."""

    required_columns = (
        CATEGORICAL_FEATURES
        + ["latitude", "longitude", "created_date"]
        + IDENTIFIER_COLUMNS
        + LEAKAGE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def create_features(df):
    """Generate features available at request creation."""

    df = df.copy()

    df["created_date"] = pd.to_datetime(
        df["created_date"],
        errors="coerce"
    )

    if df["created_date"].isna().any():
        raise ValueError(
            "Invalid or missing created_date values found."
        )

    # Extract creation-time information
    df["created_hour"] = df["created_date"].dt.hour

    df["created_day_of_week"] = (
        df["created_date"].dt.dayofweek
    )

    df["created_month"] = (
        df["created_date"].dt.month
    )

    # Treat ZIP codes as categorical values
    df["incident_zip"] = (
        df["incident_zip"]
        .astype("string")
        .replace("<NA>", pd.NA)
    )

    # Remove identifiers and post-creation information
    excluded_columns = (
        IDENTIFIER_COLUMNS
        + LEAKAGE_COLUMNS
        + ["created_date", TARGET_COLUMN]
    )

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    y = df[TARGET_COLUMN].astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError(
            "Target must contain only binary values 0 and 1."
        )

    return X, y


def build_preprocessor():
    """Build reusable numerical and categorical transformers."""

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_training_data(
    test_size=0.2,
    random_state=42
):
    """Prepare training and testing datasets."""

    config = load_config()

    df = load_processed_data(config)

    validate_features(df)

    X, y = create_features(df)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    )

    preprocessor = build_preprocessor()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


def run_feature_engineering():
    """Execute feature preparation and display a summary."""

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_training_data()

    print("\nNYC 311 FEATURE ENGINEERING")
    print("=" * 45)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print(f"Training features: {X_train.shape[1]}")
    print(f"Testing features: {X_test.shape[1]}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("\nCategorical features:")
    print(CATEGORICAL_FEATURES)

    print("\nNumerical features:")
    print(NUMERICAL_FEATURES)

    print("\nPreprocessor created successfully.")
    print("The preprocessor has not yet been fitted.")


if __name__ == "__main__":
    run_feature_engineering()
