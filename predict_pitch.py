import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/random_forest_pitch_model.joblib")


def parse_arguments():
    """Read optional command-line settings."""
    parser = argparse.ArgumentParser(
        description="Predict a baseball pitch type with a trained Random Forest model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the saved model file. Default: {MODEL_PATH}",
    )
    return parser.parse_args()


def load_model(model_path):
    """Load the saved model package from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Run 'python train_model.py' first."
        )

    return joblib.load(model_path)


def get_float_input(prompt):
    """Ask the user for a number and keep trying until the input is valid."""
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("Please enter a valid number.")


def collect_pitch_data(feature_columns):
    """Collect one pitch's feature values from the command line."""
    print("Enter the details for the new pitch.")
    print("Use the same units and scale as the training CSV.\n")

    pitch_data = {}
    for feature in feature_columns:
        friendly_name = feature.replace("_", " ")
        pitch_data[feature] = get_float_input(f"{friendly_name}: ")

    return pitch_data


def predict_pitch_type(model_package, pitch_data):
    """Predict the pitch type and probabilities for one pitch."""
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    input_df = pd.DataFrame([pitch_data], columns=feature_columns)
    predicted_pitch = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    probability_table = pd.DataFrame(
        {
            "pitch_type": model.classes_,
            "probability": probabilities,
        }
    ).sort_values(by="probability", ascending=False)

    return predicted_pitch, probability_table


def print_prediction(predicted_pitch, probability_table):
    """Print the predicted pitch type and class probabilities."""
    print("\nPrediction")
    print("----------")
    print(f"Predicted pitch type: {predicted_pitch}")

    print("\nPrediction Probabilities:")
    for _, row in probability_table.iterrows():
        print(f"{row['pitch_type']}: {row['probability']:.2%}")


def main():
    """Run the pitch prediction workflow."""
    try:
        args = parse_arguments()

        model_package = load_model(args.model)
        feature_columns = model_package["feature_columns"]

        pitch_data = collect_pitch_data(feature_columns)
        predicted_pitch, probability_table = predict_pitch_type(model_package, pitch_data)
        print_prediction(predicted_pitch, probability_table)

    except FileNotFoundError as error:
        print(f"File error: {error}")
    except KeyError as error:
        print(f"Model error: saved model is missing expected information: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    main()
