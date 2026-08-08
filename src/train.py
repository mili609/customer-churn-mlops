import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

TRAIN_PATH = "data/processed/training_processed.csv"
TEST_PATH = "data/processed/testing_processed.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load processed data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "Churn"

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("Training data:", X.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 3. Train-validation split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training split:", X_train.shape)
print("Validation split:", X_val.shape)


# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42
    )
}


# --------------------------------------------------
# 5. MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("customer-churn-prediction")


# --------------------------------------------------
# 6. Train and evaluate models
# --------------------------------------------------

results = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Validation prediction
        y_pred = model.predict(X_val)

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Log parameters
        mlflow.log_param("model_name", model_name)

        # Log metrics
        mlflow.log_metric("validation_accuracy", accuracy)
        mlflow.log_metric("validation_precision", precision)
        mlflow.log_metric("validation_recall", recall)
        mlflow.log_metric("validation_f1", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        results[model_name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


# --------------------------------------------------
# 7. Select best model
# --------------------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = results[best_model_name]["model"]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_name)
print("Validation F1:", results[best_model_name]["f1"])


# --------------------------------------------------
# 8. Evaluate ALL models on final test data
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST RESULTS - ALL MODELS")
print("=" * 60)

test_results = {}

for model_name, result in results.items():

    model = result["model"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    test_results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print("\n" + "-" * 50)
    print(model_name)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


# --------------------------------------------------
# 9. Save validation-selected model
# --------------------------------------------------

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("Validation-selected model:", best_model_name)
print("Saved to:", best_model_path)