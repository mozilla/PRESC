import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, accuracy_score

from presc.deprecated.misclassifications.misclass_rate import (
    misclass_rate_feature,
)


# --------------------
# This is no longer needed.
# It has been replaced by presc.evaluations.conditional_metric.compute_conditional_metric
# Keeping here so as not to break the tests.
# --------------------
def compute_conditional_metric(
    test_dataset,
    test_predictions,
    feature,
    categorical=False,
    bins=10,
    bins_type="regular",
    metric_function=lambda y_true, y_pred: f1_score(
        y_true, y_pred, pos_label=True, zero_division=0
    ),
):

    """Computes the conditional metrics.
    This function allows to compute the conditional metrics of a sample for
    the values of any particular feature.
    The function allows any binning for this calculation, which means that
    regularly spaced binnings, disparately spaced binnings that correspond to
    sets of an equal amount of data points (such as quartiles, deciles, or
    n-quantiles), or any other arbitrary irregular binning can be used.
    When the full dataset with all points does not have any data point in an
    interval corresponding to a certain bin, the function yields a "0" value
    for the metric.
    Parameters:
        test_dataset: Dataset with the features of all data points, where the
            true class is at the last column.
        test_predictions: List of the predicted classes for all data points.
        feature: Column name in the dataset of the feature.
        categorical (bool): Indicates whether it is a categorical feature.
            Default is "False".
        bins (int, list, str):
            * If an integer, this indicates the number of bins (default value is
            10). Whether this corresponds to dividing the feature scale in
            regularly spaced bins (default) or into quantiles, it must be
            specified in the parameter "bins_type".
            * If the string "quartiles" or "deciles" is used, then it
            automatically computes the appropriate bin edge postions to optimize
            for a quartile or decile grouping.
            * If any other feature intervals are needed, then a list of the
            feature values corresponding to the positions separating the bins
            and including the outermost edges must be provided.
        bins_type (str): If the bins parameter is an integer with the number
            of bins, this parameter allows to specify whether these bins should
            be "regular" evenly spaced bins or "quantiles". Default value is
            "regular".
        metric_function(lambda function): This parameter will take in a lambda function
            which will work on matched true y value and predicted y value. Please make sure
            to write a runnable metric score function(from sklearn.metric) with all parameters
            (with its value) you want to set. This misclass_rate_feature function will work
            exactly the metric function which you give on binned data.
            Such as: metric_function=lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label="True", zero_division=0)

    Returns:
        Two elements which correspond to:
            1) The edges of the bins in the feature scale.
            2) The list of conditional metrics in each bin.
    """
    from presc.evaluations.conditional_metric import compute_conditional_metric as ccm

    y_test_name = list(test_dataset.columns)[-1]

    y_pred_series = pd.Series(test_predictions, test_dataset.index)

    if bins == "quartiles":
        bins = 4
        bins_type = "quantile"
    elif bins == "deciles":
        bins = 10
        bins_type = "quantile"

    result = ccm(
        test_dataset[feature],
        test_dataset[y_test_name],
        y_pred_series,
        metric=metric_function,
        as_categorical=categorical,
        num_bins=bins,
        quantile=bins_type.startswith("quantile"),
    )

    return result.bins.to_numpy(), list(result.vals)


# --------------------
# This is no longer needed.
# It has been replaced by presc.evaluations.utils.get_bins
# Keeping here so as not to break the tests.
# --------------------
def compute_quantiles(dataset, feature, quantiles=4):
    """Computes optimal feature values to obtain n-quantiles.
    This function tries to determine the optimal feature value ranges in order
    to obtain groups of data of similar sizes (i.e. with an equal amount of
    samples), despite corresponding to feature intervals of different sizes.
    Very often this is not strictly possible. In particular, when the precision
    of the feature is small and many data points share the same feature values
    (i.e. the feature behaves as pseudo-discrete). In this case, these large
    subsets of data points sharing the same value either get all counted in one
    bin or they get all counted in another. Which makes it impossible to
    perfectly equilibrate the different groups.
    To arbitrarily split between two contiguous bins a subset of data points
    with the same feature value is not acceptable if different histograms and
    distributions have to be compared, or if normalization or other operations
    among them have to be carried out.
    Parameters:
        dataset (DataFrame): Data to try to chop in equal sets.
        feature: Column name in the dataset of the feature.
        quantiles (int): Number of equally-sized groups into which to try to
            divide the sample. For quartiles use 4, for deciles use 10, etc.
            Default value is 4.
    Returns:
        edge_values (list): List of the optimal edge positions.
    """
    from presc.evaluations.utils import get_bins

    return get_bins(dataset[feature], quantiles, quantile=True)[1]


@pytest.fixture
def dataset():
    dataset = pd.DataFrame(
        {
            "Feature 1": [1.0, 2.5, 3.5, 5.0],
            "Feature 2": [1.0, 6.0, 8.0, 9.0],
            "Class": ["True", "True", "False", "False"],
        },
        columns=["Feature 1", "Feature 2", "Class"],
    )
    return dataset


@pytest.fixture
def dataset_predictions():
    dataset_predictions = ["False", "False", "False", "True"]
    return dataset_predictions


@pytest.fixture(params=["Feature 1", "Feature 2"])
def feature_param(request):
    return request.param


@pytest.fixture(
    params=[
        2,
        # This case is no longer supported
        # [0.0, 4.0, 10.0],
        "quartiles",
    ]
)
def bins_param(request):
    return request.param


@pytest.fixture(params=["regular", "quantiles"])
def bins_type_param(request):
    return request.param


# --------------------
# Some of this test logic may be applicable to
# presc.evaluations.conditional_metric.compute_conditional_metric
# --------------------
def test_misclass_rate_feature_1(
    dataset, dataset_predictions, feature_param, bins_param, bins_type_param
):
    """Checks for assertions that should always apply regardless of the dataset
    and the parameters.
    """
    result_edges, result_rate, result_sd = misclass_rate_feature(
        dataset,
        dataset_predictions,
        feature=feature_param,
        bins=bins_param,
        bins_type=bins_type_param,
    )
    assert len(result_edges) == len(result_rate) + 1
    if type(bins_param) == int:
        assert len(result_rate) == bins_param
    elif type(bins_param) == list:
        assert len(result_rate) == len(bins_param) - 1
    assert max(result_rate) <= 1.0
    assert min(result_rate) >= 0.0


@pytest.fixture(
    params=[
        (
            lambda y_true, y_pred: f1_score(
                y_true, y_pred, pos_label="True", zero_division=0
            )
        ),
        (lambda y_true, y_pred: accuracy_score(y_true, y_pred)),
    ]
)
def metric_function_param(request):
    return request.param


# --------------------
# This test logic can be applied to
# presc.evaluations.conditional_metric.compute_conditional_metric
# --------------------
def test_compute_conditional_metric(
    dataset,
    dataset_predictions,
    feature_param,
    bins_param,
    bins_type_param,
    metric_function_param,
):
    """Checks for assertions that should always apply regardless of the dataset
    and the parameters.
    """
    bins, metric_list = compute_conditional_metric(
        dataset,
        dataset_predictions,
        feature=feature_param,
        bins=bins_param,
        bins_type=bins_type_param,
        metric_function=metric_function_param,
    )
    if type(bins_param) == int:
        assert len(metric_list) == bins_param
    elif type(bins_param) == list:
        assert len(bins) == len(metric_list) + 1
    assert max(metric_list) <= 1.0
    assert min(metric_list) >= 0.0


# This decorator lists de combination of parameters to be tested.
@pytest.mark.parametrize(
    "feature_scenario, "
    "categorical_scenario, bins_scenario, bins_type_scenario, "
    "result_edges_scenario, result_rate_scenario, result_sd_scenario",
    [
        (
            "Feature 1",
            False,
            2,
            "regular",
            [1.0, 3.0, 5.0],
            [1.0, 0.5],
            [1.414, 0.8535],
        ),
        (
            "Feature 1",
            False,
            2,
            "quantiles",
            [1.0, 3.0, 5.0],
            [1.0, 0.5],
            [1.414, 0.8535],
        ),
        (
            "Feature 2",
            False,
            2,
            "regular",
            [1.0, 5.0, 9.0],
            [1.0, 0.6666],
            [2.0, 0.8563],
        ),
        (
            "Feature 2",
            False,
            2,
            "quantiles",
            [1.0, 7.0, 9.0],
            [1.0, 0.5],
            [1.414, 0.8536],
        ),
        (
            "Feature 1",
            False,
            [0.0, 4.0, 10.0],
            "regular",
            [0.0, 4.0, 10.0],
            [0.6666, 1.0],
            [0.8563, 2.0],
        ),
        (
            "Feature 2",
            False,
            [0.0, 4.0, 10.0],
            "regular",
            [0.0, 4.0, 10.0],
            [1.0, 0.6666],
            [2.0, 0.8563],
        ),
        (
            "Feature 1",
            False,
            "quartiles",
            "regular",
            [1.0, 2.125, 3.0, 3.875, 5.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, float("nan"), 2.0],
        ),
        (
            "Feature 2",
            False,
            "quartiles",
            "regular",
            [1.0, 4.75, 7.0, 8.25, 9.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, float("nan"), 2.0],
        ),
    ],
)
# --------------------
# Some of these test cases may be applicable to
# presc.evaluations.conditional_metric.compute_conditional_metric
# using the misclassification rate as the metric:
# lambda y_true, y_pred: (y_true != y_pred).mean()
# The standard deviation computation is currently not supported and can be
# ignored for now.
# --------------------
def test_misclass_rate_feature_2(
    dataset,
    dataset_predictions,
    feature_scenario,
    categorical_scenario,
    bins_scenario,
    bins_type_scenario,
    result_edges_scenario,
    result_rate_scenario,
    result_sd_scenario,
):
    """Checks for assertions that should be true for specific parameter combinations."""

    result_edges, result_rate, result_sd = misclass_rate_feature(
        dataset,
        dataset_predictions,
        feature_scenario,
        categorical=categorical_scenario,
        bins=bins_scenario,
        bins_type=bins_type_scenario,
    )

    # All results should have the expected number of elements
    assert len(result_edges) == len(result_edges_scenario)
    assert len(result_rate) == len(result_rate_scenario)
    assert len(result_sd) == len(result_sd_scenario)

    # All the elements of the three results should agree with the written values
    # up to 0,1% (10^-3).
    np.testing.assert_allclose(
        result_edges, result_edges_scenario, rtol=1e-02, equal_nan=True
    )
    np.testing.assert_allclose(
        result_rate, result_rate_scenario, rtol=1e-03, equal_nan=True
    )
    np.testing.assert_allclose(
        result_sd, result_sd_scenario, rtol=1e-03, equal_nan=True
    )


@pytest.fixture
def dataset_categorical():
    dataset = pd.DataFrame(
        {
            "Feature 1": ["red", "blue", "green", "blue", "blue"],
            "Feature 2": [1.0, 6.0, 8.0, 9.0, 2.0],
            "Class": ["True", "True", "True", "False", "False"],
        },
        columns=["Feature 1", "Feature 2", "Class"],
    )
    return dataset


@pytest.fixture
def dataset_predictions_categorical():
    dataset_predictions = ["False", "False", "True", "False", "True"]
    return dataset_predictions


# --------------------
# These test cases may be applicable to
# presc.evaluations.conditional_metric.compute_conditional_metric
# using the misclassification rate as the metric:
# lambda y_true, y_pred: (y_true != y_pred).mean()
# The standard deviation computation is currently not supported and can be
# ignored for now.
# --------------------
def test_misclass_rate_feature_3(dataset_categorical, dataset_predictions_categorical):
    """Checks for assertions that should be true for specific parameter combinations."""

    result_edges, result_rate, result_sd = misclass_rate_feature(
        dataset_categorical,
        dataset_predictions_categorical,
        feature="Feature 1",
        categorical=True,
    )
    result_edges_scenario = ["red", "blue", "green"]
    result_rate_scenario = [0.6667, 0.0, 1.0]
    result_sd_scenario = [0.8563, float("nan"), 2.0]

    # All results should have the expected number of elements
    assert len(result_edges) == len(result_edges_scenario)
    assert len(result_rate) == len(result_rate_scenario)
    assert len(result_sd) == len(result_sd_scenario)

    # All the elements of the three results should agree with the written values
    # up to 0,1% (10^-3).
    np.testing.assert_array_equal(result_edges, result_edges_scenario)
    np.testing.assert_allclose(
        result_rate, result_rate_scenario, rtol=1e-03, equal_nan=True
    )
    np.testing.assert_allclose(
        result_sd, result_sd_scenario, rtol=1e-03, equal_nan=True
    )


# --------------------
# This test logic can be applied to presc.evaluations.utils.get_bins instead.
# --------------------
def test_compute_quantiles(dataset):
    selected_feature = "Feature 1"
    quantile_number = 2
    edges_bins = compute_quantiles(
        dataset, feature=selected_feature, quantiles=quantile_number
    )

    # Should always apply regardless of the dataset and the parameters.
    assert edges_bins[0] == min(dataset[selected_feature])
    assert edges_bins[-1] == max(dataset[selected_feature])
    assert len(edges_bins) == quantile_number + 1

    # True for this specific dataset
    np.testing.assert_allclose(
        edges_bins, np.asarray([1.0, 3.0, 5.0]), rtol=1e-03, equal_nan=True
    )
