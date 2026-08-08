import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocess import TARGET, preprocess_datasets


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
