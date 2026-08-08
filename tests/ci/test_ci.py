from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_required_project_directories_exist():
    for directory in ("src", "data"):
        assert (PROJECT_ROOT / directory).is_dir()


def test_requirements_file_exists():
    assert (PROJECT_ROOT / "requirements.txt").is_file()


def test_dvc_dataset_metadata_exists_without_requiring_data_download():
    raw_data = PROJECT_ROOT / "data" / "raw"
    assert (raw_data / "customer_churn_dataset-training-master.csv.dvc").is_file()
    assert (raw_data / "customer_churn_dataset-testing-master.csv.dvc").is_file()
