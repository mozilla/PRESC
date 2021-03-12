from pathlib import Path

import pytest
from pandas import read_pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from numpy import arange

from presc.model import ClassificationModel
from presc.dataset import Dataset
from presc import global_config

DATASET_DF_PATH = Path(__file__).parent / "fixtures" / "dataset.pkl"


@pytest.fixture
def dataset_df():
    return read_pickle(DATASET_DF_PATH)


@pytest.fixture
def in_test_set():
    # Deterministic train/test split for testing
    # This is used with the synthetic `dataset_df` with 100 rows
    rows = arange(100)
    return (rows < 10) | (rows >= 90)


@pytest.fixture
def train_dataset(dataset_df, in_test_set):
    return Dataset(dataset_df[~in_test_set], label_col="label")


@pytest.fixture
def test_dataset(dataset_df, in_test_set):
    return Dataset(dataset_df[in_test_set], label_col="label")


@pytest.fixture
def dataset_transform():
    return ColumnTransformer(
        [
            ("scaler", StandardScaler(), make_column_selector(dtype_include="number")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="passthrough",
    )


@pytest.fixture
def pipeline_classifier(dataset_transform):
    return Pipeline([("scaler", dataset_transform), ("clf", LogisticRegression())])


@pytest.fixture
def classification_model(pipeline_classifier, train_dataset):
    return ClassificationModel(pipeline_classifier, train_dataset, retrain_now=True)


@pytest.fixture(autouse=True)
def reset_config():
    # Reset config to default for each test
    global_config.reset_defaults()
