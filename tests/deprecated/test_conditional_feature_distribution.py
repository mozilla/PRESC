import pytest
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from presc.deprecated.conditional_feature_distribution.conditional_feature_distribution import (
    plot_all_histogram_conditional_feature_distribution,
)
from presc.deprecated.train_test_dataset import TrainTestDataset

DATASET_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/")

WINE_DATA_PATH = DATASET_DIR.joinpath("winequality.csv")
WINE_LABEL_COL = "recommend"
random_state, test_size = 0, 0.5806


@pytest.fixture
def wine_y_actual_and_y_predict_and_X_test():
    dataset = pd.read_csv(WINE_DATA_PATH)
    dataset = dataset.drop(columns=["quality"])

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = SVC(
        kernel="linear", decision_function_shape="ovr", class_weight="balanced"
    )
    classifier.fit(X_train_scaled, y_train)
    y_predict = classifier.predict(X_test_scaled)

    return (y_test, y_predict, X_test, X_train_scaled)


# test invalid feature_column
def test_wrong_type_feature_column(wine_y_actual_and_y_predict_and_X_test):
    invalid_feature_column = "abd"
    with pytest.raises(TypeError):
        plot_all_histogram_conditional_feature_distribution(
            y_actual=wine_y_actual_and_y_predict_and_X_test[0],
            y_predict=wine_y_actual_and_y_predict_and_X_test[1],
            feature_column=invalid_feature_column,
        )


def test_wrong_length_feature_column(wine_y_actual_and_y_predict_and_X_test):
    size = len(wine_y_actual_and_y_predict_and_X_test[1])
    invalid_feature_column = [1 for _ in range(size + 1)]
    with pytest.raises(ValueError):
        plot_all_histogram_conditional_feature_distribution(
            y_actual=wine_y_actual_and_y_predict_and_X_test[0],
            y_predict=wine_y_actual_and_y_predict_and_X_test[1],
            feature_column=invalid_feature_column,
        )


# test valid feature column that is numeric
def test_valid_numeric(wine_y_actual_and_y_predict_and_X_test):
    expected_group_sizes = confusion_matrix(
        wine_y_actual_and_y_predict_and_X_test[0],
        wine_y_actual_and_y_predict_and_X_test[1],
    ).ravel()
    actual_group_sizes = plot_all_histogram_conditional_feature_distribution(
        y_actual=wine_y_actual_and_y_predict_and_X_test[0],
        y_predict=wine_y_actual_and_y_predict_and_X_test[1],
        feature_column=wine_y_actual_and_y_predict_and_X_test[2].pH,
    )
    assert len(expected_group_sizes) == len(actual_group_sizes)
    assert all([a == b for a, b in zip(expected_group_sizes, actual_group_sizes)])


# test valid feature column that is categorical
def test_valid_category(wine_y_actual_and_y_predict_and_X_test):
    expected_group_sizes = confusion_matrix(
        wine_y_actual_and_y_predict_and_X_test[0],
        wine_y_actual_and_y_predict_and_X_test[1],
    ).ravel()
    actual_group_sizes = plot_all_histogram_conditional_feature_distribution(
        y_actual=wine_y_actual_and_y_predict_and_X_test[0],
        y_predict=wine_y_actual_and_y_predict_and_X_test[1],
        feature_column=[
            "1" for _ in range(len(wine_y_actual_and_y_predict_and_X_test[1]))
        ],
    )
    assert len(expected_group_sizes) == len(actual_group_sizes)
    assert all([a == b for a, b in zip(expected_group_sizes, actual_group_sizes)])


# test the feature with the TrainTestDataset API (module test)

VEHICLES_DATA_PATH = DATASET_DIR.joinpath("vehicles.csv")
VEHICLES_LABEL_COL = "Class"


@pytest.fixture
def vehicles_dataset_wrapper():
    dataset_wrapper = TrainTestDataset(
        pd.read_csv(VEHICLES_DATA_PATH), VEHICLES_LABEL_COL
    )
    dataset_wrapper.split_test_train(test_size=0.4, random_state=random_state)
    return dataset_wrapper


# test valid feature column that is numeric
def test_valid_numeric_vehicles_dw(vehicles_dataset_wrapper):

    # pre-process
    scaler = StandardScaler().fit(vehicles_dataset_wrapper.train_features)
    X_train_scaled = scaler.transform(vehicles_dataset_wrapper.train_features)
    X_test_scaled = scaler.transform(vehicles_dataset_wrapper.test_features)
    y_train = vehicles_dataset_wrapper.train_labels
    y_test = vehicles_dataset_wrapper.test_labels

    classifier = SVC(kernel="linear", decision_function_shape="ovo")
    classifier.fit(X_train_scaled, y_train)
    y_predict = classifier.predict(X_test_scaled)

    expected_group_sizes = confusion_matrix(y_test, y_predict).ravel()

    # before we consider adding the missing confusion matrix group, we will remove the 0s
    expected_group_sizes = expected_group_sizes[expected_group_sizes > 0]

    CIRCULARITY = vehicles_dataset_wrapper.test_features["CIRCULARITY"]
    actual_group_sizes = plot_all_histogram_conditional_feature_distribution(
        y_actual=y_test,
        y_predict=y_predict,
        feature_column=CIRCULARITY,
    )
    assert len(expected_group_sizes) == len(actual_group_sizes)
    assert all([a == b for a, b in zip(expected_group_sizes, actual_group_sizes)])
