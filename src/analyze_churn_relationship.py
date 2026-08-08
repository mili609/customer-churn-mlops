import pandas as pd

TRAIN_PATH = "data/raw/customer_churn_dataset-training-master.csv"
TEST_PATH = "data/raw/customer_churn_dataset-testing-master.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Remove completely empty training row
train_df = train_df.dropna(how="all")


numeric_columns = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]


print("=" * 70)
print("FEATURE MEANS BY CHURN CLASS")
print("=" * 70)


for column in numeric_columns:

    print("\n" + "-" * 60)
    print(column)

    print("\nTRAIN:")
    print(
        train_df.groupby("Churn")[column]
        .mean()
        .round(2)
    )

    print("\nTEST:")
    print(
        test_df.groupby("Churn")[column]
        .mean()
        .round(2)
    )


print("\n\n" + "=" * 70)
print("CATEGORICAL FEATURES BY CHURN")
print("=" * 70)


categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]


for column in categorical_columns:

    print("\n" + "-" * 60)
    print(column)

    print("\nTRAIN:")
    print(
        pd.crosstab(
            train_df[column],
            train_df["Churn"],
            normalize="columns"
        ).round(3)
    )

    print("\nTEST:")
    print(
        pd.crosstab(
            test_df[column],
            test_df["Churn"],
            normalize="columns"
        ).round(3)
    )