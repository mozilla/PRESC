import pytest
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from presc.conditional_feature_distribution.conditional_feature_distribution import (
    plot_all_histogram_conditional_feature_distribution,
)

DATASET_DIR = Path(__file__).resolve().parent.parent.joinpath("datasets/")

WINE_DATA_PATH = DATASET_DIR.joinpath("winequality.csv")
WINE_LABEL_COL = "recommend"
random_state = 0


@pytest.fixture
def wine_y_actual_and_y_predict_and_X_test():
    dataset = pd.read_csv(WINE_DATA_PATH)
    dataset = dataset.drop(columns=["quality"])

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5806, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = SVC(
        kernel="linear", decision_function_shape="ovr", class_weight="balanced"
    )
    classifier.fit(X_train_scaled, y_train)
    y_predict = classifier.predict(X_test_scaled)

    return (y_test, y_predict, X_test)


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


# test valid feature column
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


# test valid feature column
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
