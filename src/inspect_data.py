import pandas as pd

training_path = "data/raw/customer_churn_dataset-training-master.csv"
testing_path = "data/raw/customer_churn_dataset-testing-master.csv"

train_df = pd.read_csv(training_path)
test_df = pd.read_csv(testing_path)

print("=" * 60)
print("TRAINING DATA")
print("=" * 60)

print("Shape:", train_df.shape)

print("\nData Types:")
print(train_df.dtypes)

print("\nDuplicate Rows:", train_df.duplicated().sum())

print("\nChurn Distribution:")
print(train_df["Churn"].value_counts())

print("\nChurn Percentage:")
print(train_df["Churn"].value_counts(normalize=True) * 100)

print("\nCategorical Columns:")
for column in train_df.select_dtypes(include="object").columns:
    print(f"\n{column}:")
    print(train_df[column].value_counts())

print("\nMissing Values:")
print(train_df.isnull().sum())


print("\n\n" + "=" * 60)
print("TESTING DATA")
print("=" * 60)

print("Shape:", test_df.shape)

print("\nData Types:")
print(test_df.dtypes)

print("\nDuplicate Rows:", test_df.duplicated().sum())

print("\nChurn Distribution:")
print(test_df["Churn"].value_counts())

print("\nChurn Percentage:")
print(test_df["Churn"].value_counts(normalize=True) * 100)

print("\nCategorical Columns:")
for column in test_df.select_dtypes(include="object").columns:
    print(f"\n{column}:")
    print(test_df[column].value_counts())

print("\nMissing Values:")
print(test_df.isnull().sum())