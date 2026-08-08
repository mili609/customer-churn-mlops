import pandas as pd

TRAIN_PATH = "data/processed/training_processed.csv"
TEST_PATH = "data/processed/testing_processed.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("=" * 60)
print("PROCESSED TRAINING DATA")
print("=" * 60)

print("Shape:", train_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())

print("\nData Types:")
print(train_df.dtypes)

print("\nMissing Values:")
print(train_df.isnull().sum())

print("\nChurn Distribution:")
print(train_df["Churn"].value_counts())


print("\n\n" + "=" * 60)
print("PROCESSED TESTING DATA")
print("=" * 60)

print("Shape:", test_df.shape)

print("\nColumns:")
print(test_df.columns.tolist())

print("\nData Types:")
print(test_df.dtypes)

print("\nMissing Values:")
print(test_df.isnull().sum())

print("\nChurn Distribution:")
print(test_df["Churn"].value_counts())