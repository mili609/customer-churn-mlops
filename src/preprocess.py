import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

TRAINING_PATH = "data/raw/customer_churn_dataset-training-master.csv"
TESTING_PATH = "data/raw/customer_churn_dataset-testing-master.csv"

PROCESSED_DIR = "data/processed"

PROCESSED_TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "training_processed.csv"
)

PROCESSED_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "testing_processed.csv"
)


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAINING_PATH)
test_df = pd.read_csv(TESTING_PATH)

print("Original training shape:", train_df.shape)
print("Original testing shape:", test_df.shape)


# --------------------------------------------------
# 3. Remove completely empty rows
# --------------------------------------------------

train_df = train_df.dropna(how="all")
test_df = test_df.dropna(how="all")

print("After removing empty rows:")
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# --------------------------------------------------
# 4. Separate target variable
# --------------------------------------------------

TARGET = "Churn"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# --------------------------------------------------
# 5. Remove CustomerID
# --------------------------------------------------

X_train = X_train.drop(columns=["CustomerID"])
X_test = X_test.drop(columns=["CustomerID"])


# --------------------------------------------------
# 6. Identify column types
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# --------------------------------------------------
# 7. Numeric preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# --------------------------------------------------
# 8. Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# --------------------------------------------------
# 9. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 10. Fit on training data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformation to test data
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 11. Get feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()


# --------------------------------------------------
# 12. Convert processed data to DataFrames
# --------------------------------------------------

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# Add target column
X_train_processed[TARGET] = y_train.reset_index(drop=True)
X_test_processed[TARGET] = y_test.reset_index(drop=True)


# --------------------------------------------------
# 13. Save processed datasets
# --------------------------------------------------

os.makedirs(PROCESSED_DIR, exist_ok=True)

X_train_processed.to_csv(
    PROCESSED_TRAIN_PATH,
    index=False
)

X_test_processed.to_csv(
    PROCESSED_TEST_PATH,
    index=False
)


# --------------------------------------------------
# 14. Final output
# --------------------------------------------------

print("\nPreprocessing completed successfully!")

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)

print(
    "\nSaved training data to:",
    PROCESSED_TRAIN_PATH
)

print(
    "Saved testing data to:",
    PROCESSED_TEST_PATH
)