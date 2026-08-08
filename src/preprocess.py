import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAINING_PATH = "data/raw/customer_churn_dataset-training-master.csv"
TESTING_PATH = "data/raw/customer_churn_dataset-testing-master.csv"
PROCESSED_DIR = "data/processed"
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "training_processed.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DIR, "testing_processed.csv")
TARGET = "Churn"
ID_COLUMN = "CustomerID"


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing transformer used by the training pipeline."""
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def preprocess_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessing on training data and apply it to both datasets."""
    train_df = train_df.dropna(how="all")
    test_df = test_df.dropna(how="all")

    X_train = train_df.drop(columns=[TARGET, ID_COLUMN])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET, ID_COLUMN])
    y_test = test_df[TARGET]

    preprocessor = build_preprocessor(X_train)
    train_processed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out(),
    )
    test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out(),
    )
    train_processed[TARGET] = y_train.reset_index(drop=True)
    test_processed[TARGET] = y_test.reset_index(drop=True)

    return train_processed, test_processed, preprocessor


def main() -> None:
    train_df = pd.read_csv(TRAINING_PATH)
    test_df = pd.read_csv(TESTING_PATH)
    print("Original training shape:", train_df.shape)
    print("Original testing shape:", test_df.shape)

    train_processed, test_processed, preprocessor = preprocess_datasets(train_df, test_df)
    print("\nNumeric features:")
    print(preprocessor.transformers_[0][2])
    print("\nCategorical features:")
    print(preprocessor.transformers_[1][2])

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_processed.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_processed.to_csv(PROCESSED_TEST_PATH, index=False)

    print("\nPreprocessing completed successfully!")
    print("Processed training shape:", train_processed.shape)
    print("Processed testing shape:", test_processed.shape)
    print("\nSaved training data to:", PROCESSED_TRAIN_PATH)
    print("Saved testing data to:", PROCESSED_TEST_PATH)


if __name__ == "__main__":
    main()
