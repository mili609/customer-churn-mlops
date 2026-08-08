import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.monitor import evaluate_dataset
from src.predict import predict_customers, risk_labels
from src.preprocess import TARGET, preprocess_datasets
from src.train import RANDOM_STATE, build_models


def _sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "Age": [25.0, np.nan, 45.0, 35.0],
            "Tenure": [2, 8, 12, 4],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Contract": ["Monthly", "Annual", "Monthly", "Annual"],
            "Churn": [1, 0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "CustomerID": [5, 6],
            "Age": [30.0, np.nan],
            "Tenure": [6, 10],
            "Gender": ["Female", "Nonbinary"],
            "Contract": ["Annual", "Monthly"],
            "Churn": [0, 1],
        }
    )
    return train, test


def test_preprocessing_imputes_and_encodes_synthetic_data():
    train, test = _sample_data()

    train_processed, test_processed, preprocessor = preprocess_datasets(train, test)

    assert len(train_processed) == len(train)
    assert len(test_processed) == len(test)
    assert train_processed.isna().sum().sum() == 0
    assert test_processed.isna().sum().sum() == 0
    assert "CustomerID" not in " ".join(train_processed.columns)
    assert "numeric__Age" in train_processed.columns
    assert "numeric__Tenure" in train_processed.columns
    assert "categorical__Gender_Female" in train_processed.columns
    assert "categorical__Contract_Annual" in train_processed.columns
    assert train_processed[TARGET].tolist() == [1, 0, 1, 0]
    assert test_processed[TARGET].tolist() == [0, 1]
    assert train_processed.shape[1] == test_processed.shape[1]
    assert preprocessor is not None


def test_preprocessor_ignores_unseen_categorical_values():
    train, test = _sample_data()

    _, test_processed, _ = preprocess_datasets(train, test)

    gender_columns = [column for column in test_processed if column.startswith("categorical__Gender_")]
    assert test_processed.loc[1, gender_columns].sum() == 0


def test_lightweight_model_makes_one_prediction_per_row():
    train, test = _sample_data()
    train_processed, test_processed, _ = preprocess_datasets(train, test)
    features = train_processed.drop(columns=[TARGET])
    model = LogisticRegression(max_iter=1000, random_state=42).fit(
        features, train_processed[TARGET]
    )

    predictions = model.predict(test_processed.drop(columns=[TARGET]))

    assert predictions.shape == (len(test),)
    assert set(predictions).issubset({0, 1})


def test_batch_prediction_returns_probability_and_risk_level():
    train, test = _sample_data()
    train_processed, _, preprocessor = preprocess_datasets(train, test)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(
        train_processed.drop(columns=[TARGET]), train_processed[TARGET]
    )

    predictions = predict_customers(test.drop(columns=[TARGET]), model, preprocessor)

    assert predictions["CustomerID"].tolist() == [5, 6]
    assert predictions["churn_prediction"].shape == (len(test),)
    assert predictions["churn_probability"].between(0, 1).all()
    assert set(predictions["risk_level"]).issubset({"Low", "Medium", "High"})
    assert risk_labels(np.array([0.2, 0.4, 0.7])).tolist() == ["Low", "Medium", "High"]


def test_monitoring_evaluates_a_processed_labelled_batch():
    train, test = _sample_data()
    train_processed, test_processed, _ = preprocess_datasets(train, test)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(
        train_processed.drop(columns=[TARGET]), train_processed[TARGET]
    )

    metrics = evaluate_dataset(model, test_processed)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0 <= value <= 1 for value in metrics.values())


def test_training_models_have_explicit_reproducible_configuration():
    models = build_models()

    assert set(models) == {"Logistic Regression", "Random Forest", "HistGradient Boosting"}
    assert models["Logistic Regression"].named_steps["classifier"].random_state == RANDOM_STATE
    assert models["Random Forest"].random_state == RANDOM_STATE
    assert models["HistGradient Boosting"].random_state == RANDOM_STATE
