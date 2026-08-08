# Customer Churn MLOps

An academic MLOps project for predicting customer churn from customer profile, subscription, usage, support, payment, and interaction features. The project trains several classification models, compares them with validation metrics, evaluates them on a held-out test dataset, versions the raw datasets with DVC metadata, records experiments with MLflow, and protects the codebase with pytest and GitHub Actions.

## Business objective

The objective is to identify customers likely to churn so that a business can prioritize retention activity. `Churn` is the binary target. The project treats a higher F1 score as the model-selection criterion, balancing precision and recall for the churn class.

## MLOps objectives

This repository implements the following MLOps practices:

- **Reproducibility:** pinned Python dependencies in `requirements.txt`, fixed random state (`42`) in model training, explicit scripts, and DVC dataset metadata.
- **Version control:** source, tests, workflow configuration, and DVC pointer files are intended to be managed with Git and GitHub. Large datasets, generated data, MLflow runs, and model artifacts are excluded from Git.
- **Experiment tracking:** MLflow records a run for each trained model, including the model name, validation metrics, and a serialized model artifact.
- **Automated testing:** pytest validates preprocessing on synthetic data and performs repository/DVC-metadata sanity checks.
- **Continuous integration:** GitHub Actions installs dependencies and runs pytest on pushes and pull requests without downloading data or training models.

## Dataset

The project uses separate training and testing CSV files:

- `data/raw/customer_churn_dataset-training-master.csv`
- `data/raw/customer_churn_dataset-testing-master.csv`

The raw data contains 12 columns: `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`, and target `Churn`.

The inspected local versions contain 440,832 training rows and 64,374 testing rows after removal of completely empty rows. The external dataset source is **not documented in this repository**; this README therefore does not attribute it to a particular provider.

## Architecture and workflow

```text
DVC-tracked raw CSV metadata
          |
          v
src/inspect_data.py, src/compare_train_test.py,
src/analyze_churn_relationship.py
          |
          v
src/preprocess.py
  - drop fully empty rows
  - remove CustomerID
  - median-impute numeric columns
  - mode-impute and one-hot encode categorical columns
          |
          v
data/processed/training_processed.csv
data/processed/testing_processed.csv
          |
          v
src/train.py
  - Logistic Regression
  - Random Forest
  - HistGradient Boosting
          |
          +--> MLflow experiment: parameters, validation metrics, model artifacts
          +--> models/best_model.pkl (local generated artifact)

tests/ + pytest.ini --> GitHub Actions CI
```

## Repository layout

```text
customer-churn-mlops/
├── .dvc/                         # DVC configuration
├── .github/workflows/ci.yml      # GitHub Actions test workflow
├── data/
│   ├── raw/
│   │   ├── customer_churn_dataset-training-master.csv.dvc
│   │   └── customer_churn_dataset-testing-master.csv.dvc
│   └── processed/                # generated, Git-ignored outputs
├── models/                       # generated, Git-ignored model artifacts
├── src/
│   ├── inspect_data.py
│   ├── preprocess.py
│   ├── inspect_processed.py
│   ├── compare_train_test.py
│   ├── analyze_churn_relationship.py
│   └── train.py
├── tests/
│   ├── test_pipeline.py
│   └── ci/test_ci.py
├── pytest.ini                    # confines discovery to tests/
├── requirements.txt
└── README.md
```

## Data ingestion and inspection

The scripts read CSV inputs with pandas.

- `src/inspect_data.py` reports shape, dtypes, duplicates, missing values, target distribution, and categorical-value distributions for both raw datasets.
- `src/compare_train_test.py` compares numeric summary statistics, categorical distributions, and churn prevalence between train and test.
- `src/analyze_churn_relationship.py` reports numeric feature means by churn class and categorical churn cross-tabs.
- `src/inspect_processed.py` checks the generated processed datasets, including shapes, columns, types, missing values, and churn distribution.

Run an inspection script from the repository root, for example:

```powershell
.\.venv\Scripts\python.exe src\inspect_data.py
```

These analysis scripts require the raw or processed local data, so they are deliberately not run by CI.

## Preprocessing

`src/preprocess.py` fits preprocessing on training features and applies the same fitted transformer to the testing features:

1. Removes completely empty rows.
2. Separates `Churn` as the target and removes `CustomerID` from features.
3. Identifies `int64`/`float64` columns as numeric and `object` columns as categorical.
4. Median-imputes numeric features.
5. Most-frequent-imputes categorical features, then applies `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
6. Writes transformed train and test data with `Churn` restored to `data/processed/`.

The module also exposes `build_preprocessor()` and `preprocess_datasets()` so the same production logic can be tested with in-memory synthetic data.

## Model training and evaluation

`src/train.py` reads the processed datasets, creates a stratified 80/20 train-validation split of the processed training data (`random_state=42`), and trains:

- Logistic Regression (`max_iter=1000`)
- Random Forest (`n_estimators=100`, `n_jobs=-1`)
- HistGradient Boosting (`max_iter=100`)

For each model, it calculates validation accuracy, precision, recall, and F1. The model with the largest validation F1 is selected. All three models are then evaluated against the separate processed testing dataset; the selected model is saved locally as `models/best_model.pkl`.

### Validation versus test performance

The validation split is sampled from the training dataset, so it has the same distribution as the data used to fit the preprocessing transformer and models. The final test set is a separate distribution and shows a substantial observed shift:

| Measure | Training | Testing |
|---|---:|---:|
| Churn rate | 56.71% | 47.37% |
| Mean age | 39.37 | 41.97 |
| Mean support calls | 3.60 | 5.40 |
| Mean payment delay | 12.97 | 17.13 |
| Mean total spend | 631.62 | 541.02 |
| Monthly contract share | 19.76% | 34.38% |

This train/test distribution shift explains why very strong validation metrics do not transfer directly to the held-out test set. For example, the saved selected HistGradient Boosting model has a validation F1 near 1.0 but a test F1 of 0.6558. This is a generalization concern rather than evidence that the validation calculation alone is sufficient for deployment decisions.

### Final results

The validation metrics below are the latest metrics recorded in local MLflow runs. The final-test row was calculated by scoring the locally saved `models/best_model.pkl` against the processed testing file. The current training script prints final-test metrics for all models but does not log those metrics to MLflow; therefore final-test values for the two non-selected models are not claimed here.

| Model | Evaluation split | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Validation | 0.8933 | 0.9232 | 0.8854 | 0.9039 |
| Random Forest | Validation | 0.9993 | 0.9999 | 0.9988 | 0.9993 |
| HistGradient Boosting | Validation | 0.9999 | 1.0000 | 0.9998 | 0.9999 |
| HistGradient Boosting (saved best model) | Final test | 0.5034 | 0.4882 | 0.9988 | 0.6558 |

## DVC data tracking

DVC metadata tracks the two raw datasets through:

- `data/raw/customer_churn_dataset-training-master.csv.dvc`
- `data/raw/customer_churn_dataset-testing-master.csv.dvc`

The metadata records checksums and file sizes, while the large `.csv` contents remain Git-ignored. This is metadata tracking only: **no DVC remote is configured in the current repository** (`dvc remote list` is empty). Consequently, another developer needs access to the dataset files or must configure an appropriate DVC remote before `dvc pull` can retrieve data.

Useful Windows commands:

```powershell
# Check tracked data state
.\.venv\Scripts\dvc.exe status

# List configured remotes (currently empty)
.\.venv\Scripts\dvc.exe remote list

# Use only after a maintainer has configured a remote with access to the data
.\.venv\Scripts\dvc.exe pull
```

## MLflow experiment tracking

Training uses the `customer-churn-prediction` experiment. Each model run records:

- Parameter: `model_name`
- Metrics: `validation_accuracy`, `validation_precision`, `validation_recall`, and `validation_f1`
- Artifact: the trained scikit-learn model via `mlflow.sklearn.log_model()`

The local workspace currently contains `mlflow.db` and `mlruns/`, both Git-ignored generated state. Start the UI locally with:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

MLflow is used during local training only; the CI workflow does not start MLflow.

## Testing and continuous integration

`pytest.ini` sets `testpaths = tests`, preventing analysis scripts such as `src/compare_train_test.py` from being collected as tests.

- `tests/test_pipeline.py` uses small synthetic DataFrames to test median imputation, categorical encoding, unknown-category handling, target retention, and prediction output shape.
- `tests/ci/test_ci.py` verifies tracked repository structure, `requirements.txt`, and the two DVC metadata files. It does not require DVC data, processed datasets, models, MLflow, or network access.

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on pushes and pull requests. It uses Python 3.13, installs `requirements.txt` and pytest, then runs `python -m pytest`. It does not run `src/train.py`, download DVC data, start MLflow, or require model artifacts.

## Windows setup and reproducibility

### Prerequisites

- Windows with Python 3.13 installed and available as `py -3.13`
- Git
- Access to the DVC-managed raw datasets, or a DVC remote configured by a maintainer

### Environment setup

From PowerShell at the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use a session-scoped policy change:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Reproduce the local data and model workflow

1. Ensure the raw CSVs are present in `data/raw/`. With an approved configured DVC remote, run `dvc pull`; otherwise obtain the matching files from the project maintainer.
2. Optionally inspect the raw data:

   ```powershell
   python src\inspect_data.py
   python src\compare_train_test.py
   python src\analyze_churn_relationship.py
   ```

3. Build processed datasets:

   ```powershell
   python src\preprocess.py
   ```

4. Optionally inspect processed outputs:

   ```powershell
   python src\inspect_processed.py
   ```

5. Train, evaluate, log MLflow runs, and save the selected model:

   ```powershell
   python src\train.py
   ```

6. Run automated tests:

   ```powershell
   pytest
   pytest tests\ci
   ```

Git and GitHub should version source code, tests, workflow files, DVC metadata, and documentation. Generated artifacts such as raw/processed CSV contents, `models/`, `mlflow.db`, and `mlruns/` intentionally remain outside Git. The current implementation does not include Docker, an API service, cloud deployment, or monitoring dashboards.
