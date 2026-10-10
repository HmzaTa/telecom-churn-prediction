"""
Shared test fixtures.

The tests never need the Kaggle dataset or the real trained model. They build a
small model from the project's own synthetic-data generator, so they run
anywhere (including CI) in a few seconds.
"""

import sys
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data_preprocessing import download_sample_data, full_preprocessing_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def model_bundle(tmp_path_factory):
    """A real model bundle {model, scaler, feature_names} trained on synthetic data."""
    tmp = tmp_path_factory.mktemp("churn")
    raw_path = tmp / "raw" / "telco.csv"
    download_sample_data(str(raw_path))

    X_train, _, y_train, _, scaler, feature_names = full_preprocessing_pipeline(
        str(raw_path), processed_dir=str(tmp / "processed")
    )
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    return {"model": model, "scaler": scaler, "feature_names": feature_names}
