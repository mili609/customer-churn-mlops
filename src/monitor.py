"""Lightweight batch evaluation for tracking selected-model performance over time."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

try:
    from src.train import EXPERIMENT_NAME, MODEL_PATH, TARGET, classification_metrics
except ModuleNotFoundError:  # Allows `python src/monitor.py` from the repository root.
    from train import EXPERIMENT_NAME, MODEL_PATH, TARGET, classification_metrics


def evaluate_dataset(model: Any, evaluation_df: pd.DataFrame) -> dict[str, float]:
    """Evaluate a fitted model on a processed, labelled batch dataset."""
    if TARGET not in evaluation_df:
        raise ValueError(f"Evaluation data must include target column '{TARGET}'.")
    return classification_metrics(
        evaluation_df[TARGET], model.predict(evaluation_df.drop(columns=[TARGET]))
    )


def log_evaluation(metrics: dict[str, float], dataset_path: str) -> None:
    """Create an MLflow batch-evaluation run for a labelled processed dataset."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    previous_f1 = None
    if experiment is not None:
        previous_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.run_type = 'batch_evaluation'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not previous_runs.empty and "metrics.evaluation_f1" in previous_runs:
            previous_f1 = previous_runs.iloc[0]["metrics.evaluation_f1"]

    with mlflow.start_run(run_name="batch-evaluation"):
        mlflow.set_tag("run_type", "batch_evaluation")
        mlflow.log_param("evaluation_dataset", dataset_path)
        mlflow.log_metrics({f"evaluation_{name}": value for name, value in metrics.items()})
        if previous_f1 is not None:
            mlflow.log_metric("evaluation_f1_change_from_previous", metrics["f1"] - previous_f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the selected model on a labelled processed CSV.")
    parser.add_argument("--data", required=True, help="Processed CSV containing the Churn target.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the selected model artifact.")
    parser.add_argument("--no-log", action="store_true", help="Calculate metrics without creating an MLflow run.")
    args = parser.parse_args()

    metrics = evaluate_dataset(joblib.load(args.model), pd.read_csv(args.data))
    print(metrics)
    if not args.no_log:
        log_evaluation(metrics, args.data)


if __name__ == "__main__":
    main()
