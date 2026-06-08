import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Update this list if your dataset uses more, fewer, or differently named features.
FEATURE_COLUMNS = [
    "velocity",
    "spin_rate",
    "vertical_break",
    "horizontal_break",
    "release_height",
    "release_side",
    "extension",
    "spin_axis",
]
TARGET_COLUMN = "pitch_type"

DATA_PATH = Path("data/sample_pitch_data.csv")
MODEL_PATH = Path("models/random_forest_pitch_model.joblib")
TEST_SIZE = 0.25
RANDOM_STATE = 42


def parse_arguments():
    """Read optional command-line settings."""
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model to classify baseball pitch types."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help=f"Path to the pitch data CSV file. Default: {DATA_PATH}",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Where to save the trained model. Default: {MODEL_PATH}",
    )
    return parser.parse_args()


def load_data(csv_path):
    """Load pitch data from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find the CSV file: {csv_path}")

    return pd.read_csv(csv_path)


def check_required_columns(data, feature_columns, target_column):
    """Make sure the dataset contains every feature column and the target column."""
    required_columns = feature_columns + [target_column]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"The dataset is missing these required columns: {missing_text}")


def prepare_data(data, feature_columns, target_column):
    """Select model columns, remove missing values, and split features from labels."""
    model_data = data[feature_columns + [target_column]].copy()

    rows_before = len(model_data)
    model_data = model_data.dropna()
    rows_after = len(model_data)
    dropped_rows = rows_before - rows_after

    if dropped_rows > 0:
        print(f"Removed {dropped_rows} rows with missing values.")

    if model_data.empty:
        raise ValueError("No usable rows remain after removing missing values.")

    X = model_data[feature_columns]
    y = model_data[target_column]
    return X, y


def check_target_classes(y):
    """Confirm the target labels can be split safely into train and test sets."""
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two pitch types.")

    class_counts = y.value_counts()
    smallest_class_count = class_counts.min()
    if smallest_class_count < 2:
        rare_classes = class_counts[class_counts < 2].index.tolist()
        raise ValueError(
            "Each pitch type needs at least two rows for a stratified train/test split. "
            f"Pitch types with too few rows: {rare_classes}"
        )


def train_random_forest(X_train, y_train):
    """Create and train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print common classification metrics for the trained model."""
    predictions = model.predict(X_test)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)

    print("Confusion Matrix:")
    print(confusion_df)


def print_feature_importance(model, feature_columns):
    """Show which features the model used most when making decisions."""
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    print("\nMost Important Features")
    print("-----------------------")
    for _, row in importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")


def save_model(model, model_path, feature_columns, target_column):
    """Save the trained model and metadata needed by the prediction script."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": target_column,
    }
    joblib.dump(model_package, model_path)
    print(f"\nSaved trained model to: {model_path}")


def main():
    """Run the full model training workflow."""
    try:
        args = parse_arguments()

        data = load_data(args.data)
        check_required_columns(data, FEATURE_COLUMNS, TARGET_COLUMN)
        X, y = prepare_data(data, FEATURE_COLUMNS, TARGET_COLUMN)
        check_target_classes(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        model = train_random_forest(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        print_feature_importance(model, FEATURE_COLUMNS)
        save_model(model, args.model, FEATURE_COLUMNS, TARGET_COLUMN)

    except FileNotFoundError as error:
        print(f"File error: {error}")
    except ValueError as error:
        print(f"Data error: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    main()
