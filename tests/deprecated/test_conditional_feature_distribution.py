import pytest
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from presc.deprecated.train_test_dataset import TrainTestDataset

DATASET_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/")

WINE_DATA_PATH = DATASET_DIR.joinpath("winequality.csv")
WINE_LABEL_COL = "recommend"
random_state, test_size = 0, 0.5806


# --------------------
# This is no longer needed.
# It has been replaced by presc.evaluations.conditional_distribution
# Keeping here with modifications so as not to break the tests.
# --------------------
def plot_all_histogram_conditional_feature_distribution(
    y_predict, feature_column, y_actual, bins: int = None
):
    """
    Description:
        Plots histograms that represens the distribution of a feature.
        Each histogram should corresponds to a group on the confusion matrix of the test data.

    Output:
        X-axis represents the bins over the range of column values for that feature.
        Y-axis represents raw frequency counts.
        If categorical, then the x-axis would be categories instead of bins.

    Args:
        y_predict (pd.DataFrame): Predicted labels for a test set from a trained model
        feature_column (object): Column of data from the test set
        y_actual (pd.DataFrame): True labels for a test set from a trained model
        bins (int): Number of bins for the histograms. Defaults to None
    """
    # TODO: plot the histograms all on the same page in the order as the cofusion matrix
    # check if the feature_column is either a pandas series of list
    if not isinstance(feature_column, (pd.Series, list)):
        raise (
            TypeError(
                "The feature column should be either a list or a pandas Series object"
            )
        )

    # check if feature_column is the same length as the dataset length
    if len(feature_column) != y_actual.size or y_predict.size != y_actual.size:
        raise (
            ValueError(
                "The feature column should have the same length as the dataset and the predicted labels"
            )
        )

    from presc.evaluations.conditional_distribution import (
        compute_conditional_distribution as ccd,
    )

    # compute_conditional_distribution expects all inputs to be Series with
    # aligned indexes.
    feature_column = pd.Series(feature_column)
    y_actual = pd.Series(y_actual).set_axis(feature_column.index)
    y_predict = pd.Series(y_predict).set_axis(feature_column.index)
    result = ccd(
        feature_column, y_actual, y_predict, binning=bins or "fd", common_bins=False
    )

    try:
        xlab = feature_column.name
    except AttributeError:
        xlab = None
    xlab = xlab or "feature"
    result.display_result(xlab)

    confusion_matrix_group = result.vals.index.map(
        lambda x: f"{x[0]}_predicted_as_{x[1]}"
    )
    return result.vals.groupby(confusion_matrix_group).sum()


@pytest.fixture
# --------------------
# Test case based on dataset in the repo.
# Not sure if we want to keep this
# --------------------
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
# --------------------
# Input validity checking is not yet done in compute_conditional_distribution
# Should be added as a part of Issue #248
# --------------------
def test_wrong_type_feature_column(wine_y_actual_and_y_predict_and_X_test):
    invalid_feature_column = "abd"
    with pytest.raises(TypeError):
        plot_all_histogram_conditional_feature_distribution(
            y_actual=wine_y_actual_and_y_predict_and_X_test[0],
            y_predict=wine_y_actual_and_y_predict_and_X_test[1],
            feature_column=invalid_feature_column,
        )


# --------------------
# Input validity checking is not yet done in compute_conditional_distribution
# Should be added as a part of Issue #248
# --------------------
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
# --------------------
# Could be used as a part of testing for compute_conditional_distribution
# This just checks that the grouping by expected/predicted labels worked,
# but not the distributions themselves.
# --------------------
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
# --------------------
# Could be used as a part of testing for compute_conditional_distribution
# This just checks that the grouping by expected/predicted labels worked,
# but not the distributions themselves.
# This covers the case of a categorical with a single value
# (ie. generating a single bin).
# --------------------
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
# --------------------
# Similar as above, using a different dataset.
# Probably don't need this
# --------------------
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
