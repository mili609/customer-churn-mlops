import pandas as pd

TRAIN_PATH = "data/raw/customer_churn_dataset-training-master.csv"
TEST_PATH = "data/raw/customer_churn_dataset-testing-master.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Remove completely empty rows
train_df = train_df.dropna(how="all")

print("=" * 60)
print("TRAIN vs TEST FEATURE DISTRIBUTION")
print("=" * 60)

numeric_columns = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

for column in numeric_columns:

    print("\n" + "-" * 50)
    print(column)

    print(
        "TRAIN -> "
        f"mean={train_df[column].mean():.2f}, "
        f"std={train_df[column].std():.2f}, "
        f"min={train_df[column].min():.2f}, "
        f"max={train_df[column].max():.2f}"
    )

    print(
        "TEST  -> "
        f"mean={test_df[column].mean():.2f}, "
        f"std={test_df[column].std():.2f}, "
        f"min={test_df[column].min():.2f}, "
        f"max={test_df[column].max():.2f}"
    )


print("\n" + "=" * 60)
print("CATEGORICAL DISTRIBUTIONS")
print("=" * 60)

categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

for column in categorical_columns:

    print("\n" + "-" * 50)
    print(column)

    print("\nTRAIN:")
    print(
        train_df[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nTEST:")
    print(
        test_df[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nTRAIN:")
print(
    train_df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTEST:")
print(
    test_df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)