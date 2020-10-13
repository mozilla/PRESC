import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from presc.dataset_wrapper import DatasetWrapper

wine_data_path = "../datasets/winequality.csv"
random_state, test_size = 0, 0.2

# test invalid initailization of DatasetWrapper object


def test_fie_does_not_exist():
    with pytest.raises(FileNotFoundError):
        DatasetWrapper("datasets/winequalit.csv")


def test_fie_format_incorrect():
    with pytest.raises(IOError):
        DatasetWrapper("datasets/README.md")


# test winequality dataset and functionalities:
@pytest.fixture
def wine_dataset_wrapper():
    dataset_wrapper = DatasetWrapper(wine_data_path)
    return dataset_wrapper


@pytest.fixture
def expected_wine_dataset():
    expected_dataset = pd.read_csv(wine_data_path)
    return expected_dataset


def test_get_raw_dataset(wine_dataset_wrapper, expected_wine_dataset):
    actual_dataset = wine_dataset_wrapper.get_raw_dataset()
    assert actual_dataset.equals(expected_wine_dataset)


def test_get_features(wine_dataset_wrapper, expected_wine_dataset):
    actual_X = wine_dataset_wrapper.get_features()
    assert actual_X.equals(expected_wine_dataset.iloc[:, :-1])


def test_get_label(wine_dataset_wrapper, expected_wine_dataset):
    actual_y = wine_dataset_wrapper.get_label()
    assert actual_y.equals(expected_wine_dataset["recommend"])


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


# test invalid subset for get_features and get_label
def test_invalid_get_features(wine_dataset_wrapper):
    with pytest.raises(ValueError):
        wine_dataset_wrapper.get_features(subset="hello")


def test_invalid_get_label(wine_dataset_wrapper):
    with pytest.raises(ValueError):
        wine_dataset_wrapper.get_label(subset="wine")
