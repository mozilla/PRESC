import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from sklearn.model_selection import train_test_split
from pathlib import Path
from presc.deprecated.train_test_dataset import TrainTestDataset

DATASET_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/")

WINE_DATA_PATH = DATASET_DIR.joinpath("winequality.csv")
WINE_LABEL_COL = "recommend"
random_state, test_size = 0, 0.2

# test invalid initailization of Dataset object


def test_label_not_in_dataset(expected_wine_dataset):
    with pytest.raises(KeyError):
        TrainTestDataset(expected_wine_dataset, "wrong_label")


# test winequality dataset and functionalities:
@pytest.fixture
def wine_dataset_wrapper(expected_wine_dataset):
    dataset_wrapper = TrainTestDataset(expected_wine_dataset, WINE_LABEL_COL)
    return dataset_wrapper


@pytest.fixture
def expected_wine_dataset():
    expected_dataset = pd.read_csv(WINE_DATA_PATH)
    return expected_dataset


def test_get_raw_dataset(wine_dataset_wrapper, expected_wine_dataset):
    actual_dataset = wine_dataset_wrapper.raw_dataset
    assert actual_dataset.equals(expected_wine_dataset)


def test_get_features(wine_dataset_wrapper, expected_wine_dataset):
    expected_feats = expected_wine_dataset.drop(columns=[WINE_LABEL_COL])
    actual_X = wine_dataset_wrapper.features
    assert actual_X.equals(expected_feats)
    assert_array_equal(wine_dataset_wrapper.feature_names, expected_feats.columns)


def test_get_label(wine_dataset_wrapper, expected_wine_dataset):
    actual_y = wine_dataset_wrapper.labels
    assert actual_y.equals(expected_wine_dataset[WINE_LABEL_COL])


def test_split_test_train(wine_dataset_wrapper, expected_wine_dataset):
    wine_dataset_wrapper.split_test_train(
        test_size=test_size, random_state=random_state
    )

    (
        expected_train_X,
        expected_test_X,
        expected_train_y,
        expected_test_y,
    ) = train_test_split(
        expected_wine_dataset.iloc[:, :-1],
        expected_wine_dataset["recommend"],
        random_state=random_state,
        test_size=test_size,
    )

    expected_train = expected_train_X.merge(
        expected_train_y, left_index=True, right_index=True, how="left"
    )
    expected_test = expected_test_X.merge(
        expected_test_y, left_index=True, right_index=True, how="left"
    )

    assert expected_train_X.equals(wine_dataset_wrapper.train_features)
    assert expected_test_X.equals(wine_dataset_wrapper.test_features)
    assert expected_train_y.equals(wine_dataset_wrapper.train_labels)
    assert expected_test_y.equals(wine_dataset_wrapper.test_labels)

    assert expected_train.equals(wine_dataset_wrapper.get_train_dataset())
    assert expected_test.equals(wine_dataset_wrapper.get_test_dataset())


def test_set_label(wine_dataset_wrapper, expected_wine_dataset):
    wine_dataset_wrapper.set_label("quality")
    assert wine_dataset_wrapper._y.name == "quality"
    assert wine_dataset_wrapper.labels.equals(expected_wine_dataset["quality"])
    assert wine_dataset_wrapper.features.equals(
        expected_wine_dataset.drop(columns=["quality"])
    )


def test_set_label_not_exist(wine_dataset_wrapper):
    with pytest.raises(KeyError):
        wine_dataset_wrapper.set_label("not_a_label")
