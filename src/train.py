"""Train and compare churn models using reproducible validation and test splits."""

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = "data/processed/training_processed.csv"
TEST_PATH = "data/processed/testing_processed.csv"
MODEL_PATH = "models/best_model.pkl"
TARGET = "Churn"
EXPERIMENT_NAME = "customer-churn-prediction"
RANDOM_STATE = 42


def build_models() -> dict[str, Any]:
    """Return the project's explicitly configured candidate models."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGradient Boosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=RANDOM_STATE
        ),
    }


def classification_metrics(y_true: pd.Series, predictions: Any) -> dict[str, float]:
    """Calculate the common binary-classification metrics used by this project."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def _mlflow_parameters(model_name: str, model: Any) -> dict[str, str]:
    parameters = {
        "model_name": model_name,
        "random_state": str(RANDOM_STATE),
        "preprocessing": json.dumps(
            {
                "id_column_removed": "CustomerID",
                "numeric_imputer": "median",
                "categorical_imputer": "most_frequent",
                "categorical_encoder": "one_hot_handle_unknown_ignore",
            },
            sort_keys=True,
        ),
    }
    parameters.update({f"model__{key}": str(value) for key, value in model.get_params().items()})
    return parameters


def train_and_evaluate(
    train_df: pd.DataFrame, test_df: pd.DataFrame, log_to_mlflow: bool = True
) -> tuple[str, Any, dict[str, dict[str, dict[str, float]]]]:
    """Fit candidates, select by validation F1 only, and evaluate once on test data."""
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    if log_to_mlflow:
        mlflow.set_experiment(EXPERIMENT_NAME)

    results: dict[str, dict[str, dict[str, float]]] = {}
    fitted_models: dict[str, Any] = {}
    for model_name, model in build_models().items():
        if log_to_mlflow:
            run_context = mlflow.start_run(run_name=model_name)
        else:
            run_context = None

        try:
            if run_context:
                run_context.__enter__()
                mlflow.log_params(_mlflow_parameters(model_name, model))

            model.fit(X_train, y_train)
            validation_metrics = classification_metrics(y_val, model.predict(X_val))
            test_metrics = classification_metrics(y_test, model.predict(X_test))
            if run_context:
                mlflow.log_metrics(
                    {f"validation_{name}": value for name, value in validation_metrics.items()}
                )
                mlflow.log_metrics({f"test_{name}": value for name, value in test_metrics.items()})
                mlflow.sklearn.log_model(sk_model=model, name="model")

            results[model_name] = {"validation": validation_metrics, "test": test_metrics}
            fitted_models[model_name] = model
        finally:
            if run_context:
                run_context.__exit__(None, None, None)

    best_model_name = max(results, key=lambda name: results[name]["validation"]["f1"])
    return best_model_name, fitted_models[best_model_name], results


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    best_model_name, best_model, results = train_and_evaluate(train_df, test_df)

    print("\nMODEL RESULTS")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for split, split_metrics in metrics.items():
            print(f"{split.title()}: {split_metrics}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nValidation-selected model: {best_model_name}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
