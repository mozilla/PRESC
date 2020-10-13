import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from pathlib import Path
from presc.dataset_wrapper import DatasetWrapper

DATASET_DIR = Path(__file__).resolve().parent.parent.joinpath("datasets/")

WINE_DATA_PATH = DATASET_DIR.joinpath("winequality.csv")
WINE_LABEL_COL = "recommend"
random_state, test_size = 0, 0.2

# test invalid initailization of DatasetWrapper object


def test_fie_does_not_exist():
    with pytest.raises(FileNotFoundError):
        DatasetWrapper("does_not_exist.csv", "trivial")


def test_fie_format_incorrect():
    with pytest.raises(pd.errors.ParserError):
        DatasetWrapper(Path(__file__).parent.parent.joinpath("setup.py"), "trivial")


def test_label_not_in_dataset():
    with pytest.raises(KeyError):
        DatasetWrapper(WINE_DATA_PATH, "wrong_label")


# test winequality dataset and functionalities:
@pytest.fixture
def wine_dataset_wrapper():
    dataset_wrapper = DatasetWrapper(WINE_DATA_PATH, WINE_LABEL_COL)
    return dataset_wrapper


@pytest.fixture
def expected_wine_dataset():
    expected_dataset = pd.read_csv(WINE_DATA_PATH)
    return expected_dataset


def test_get_raw_dataset(wine_dataset_wrapper, expected_wine_dataset):
    actual_dataset = wine_dataset_wrapper.get_raw_dataset()
    assert actual_dataset.equals(expected_wine_dataset)


def test_get_features(wine_dataset_wrapper, expected_wine_dataset):
    actual_X = wine_dataset_wrapper.get_features()
    assert actual_X.equals(expected_wine_dataset.drop(columns=[WINE_LABEL_COL]))


def test_get_label(wine_dataset_wrapper, expected_wine_dataset):
    actual_y = wine_dataset_wrapper.get_label()
    assert actual_y.equals(expected_wine_dataset[WINE_LABEL_COL])


def test_split_test_train(wine_dataset_wrapper, expected_wine_dataset):
    wine_dataset_wrapper.split_test_train()

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

    assert expected_train_X.equals(wine_dataset_wrapper.get_features(subset="train"))
    assert expected_test_X.equals(wine_dataset_wrapper.get_features(subset="test"))
    assert expected_train_y.equals(wine_dataset_wrapper.get_label(subset="train"))
    assert expected_test_y.equals(wine_dataset_wrapper.get_label(subset="test"))

    assert expected_train.equals(wine_dataset_wrapper.get_train_dataset())
    assert expected_test.equals(wine_dataset_wrapper.get_test_dataset())


def test_set_label(wine_dataset_wrapper, expected_wine_dataset):
    assert wine_dataset_wrapper.get_label().equals(
        expected_wine_dataset[WINE_LABEL_COL]
    )

    wine_dataset_wrapper.set_label("quality")
    assert wine_dataset_wrapper.y.name == "quality"
    assert wine_dataset_wrapper.get_label().equals(expected_wine_dataset["quality"])
    assert wine_dataset_wrapper.get_features().equals(
        expected_wine_dataset.drop(columns=["quality"])
    )


def test_set_label_not_exist(wine_dataset_wrapper):
    with pytest.raises(KeyError):
        wine_dataset_wrapper.set_label("not_a_label")


# test invalid subset for get_features and get_label
def test_invalid_get_features(wine_dataset_wrapper):
    with pytest.raises(ValueError):
        wine_dataset_wrapper.get_features(subset="hello")


def test_invalid_get_label(wine_dataset_wrapper):
    with pytest.raises(ValueError):
        wine_dataset_wrapper.get_label(subset="wine")
