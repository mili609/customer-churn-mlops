"""Batch churn prediction and risk classification for new customer records."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from src.preprocess import ID_COLUMN, PREPROCESSOR_PATH, TARGET
except ModuleNotFoundError:  # Allows `python src/predict.py` from the repository root.
    from preprocess import ID_COLUMN, PREPROCESSOR_PATH, TARGET


MODEL_PATH = "models/best_model.pkl"
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40


def risk_labels(probabilities: np.ndarray) -> np.ndarray:
    """Map churn probabilities to transparent retention-priority bands."""
    return np.select(
        [probabilities >= HIGH_RISK_THRESHOLD, probabilities >= MEDIUM_RISK_THRESHOLD],
        ["High", "Medium"],
        default="Low",
    )


def predict_customers(
    customers: pd.DataFrame, model: Any, preprocessor: Any
) -> pd.DataFrame:
    """Return churn predictions, positive-class probabilities, and risk labels."""
    features = customers.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    transformed = pd.DataFrame(
        preprocessor.transform(features), columns=preprocessor.get_feature_names_out()
    )
    probabilities = model.predict_proba(transformed)
    positive_index = list(model.classes_).index(1)
    churn_probability = probabilities[:, positive_index]

    result = pd.DataFrame(index=customers.index)
    if ID_COLUMN in customers:
        result[ID_COLUMN] = customers[ID_COLUMN]
    result["churn_prediction"] = model.predict(transformed)
    result["churn_probability"] = churn_probability
    result["risk_level"] = risk_labels(churn_probability)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate batch churn-risk predictions.")
    parser.add_argument("--input", required=True, help="CSV containing raw customer features.")
    parser.add_argument("--output", required=True, help="Path for prediction CSV output.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to a trained model artifact.")
    parser.add_argument(
        "--preprocessor", default=PREPROCESSOR_PATH, help="Path to the fitted preprocessing artifact."
    )
    args = parser.parse_args()

    predictions = predict_customers(
        pd.read_csv(args.input), joblib.load(args.model), joblib.load(args.preprocessor)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
