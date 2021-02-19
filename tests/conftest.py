from pathlib import Path

import pytest
from pandas import read_pickle

DATASET_DF_PATH = Path(__file__).parent / "fixtures" / "dataset.pkl"


@pytest.fixture
def dataset_df():
    return read_pickle(DATASET_DF_PATH)
