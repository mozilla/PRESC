import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, accuracy_score

from presc.deprecated.misclassifications.misclass_rate import (
    misclass_rate_feature,
    show_misclass_rate_feature,
    show_misclass_rates_features,
    compute_quantiles,
    show_quantiles_feature,
    show_quantiles_features,
    compute_conditional_metric,
    feature_binning,
)


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


@pytest.fixture(params=[2, [0.0, 4.0, 10.0], "quartiles"])
def bins_param(request):
    return request.param


@pytest.fixture(params=["regular", "quantiles"])
def bins_type_param(request):
    return request.param


def test1_misclass_rate_feature(
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


def test1_compute_conditional_metric(
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


def test_feature_binning(dataset, feature_param, bins_param, bins_type_param):
    bins = feature_binning(
        dataset,
        feature=feature_param,
        bins=bins_param,
        bins_type=bins_type_param,
    )
    if type(bins_param) == int:
        assert len(bins) == bins_param + 1
    elif type(bins_param) == list:
        assert len(bins) == len(bins_param)


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
def test2_misclass_rate_feature(
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
        result_edges, result_edges_scenario, rtol=1e-03, equal_nan=True
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


def test3_misclass_rate_feature(dataset_categorical, dataset_predictions_categorical):
    """Checks for assertions that should be true for specific parameter combinations."""

    result_edges, result_rate, result_sd = misclass_rate_feature(
        dataset_categorical,
        dataset_predictions_categorical,
        feature="Feature 1",
        categorical=True,
    )
    result_edges_scenario = ["blue", "green", "red"]
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


def test_show_misclass_rate_feature(dataset, dataset_predictions):
    show_misclass_rate_feature(dataset, dataset_predictions, "Feature 2")


def test_show_misclass_rates_features(dataset, dataset_predictions):
    show_misclass_rates_features(dataset, dataset_predictions)


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


def test_show_quantiles_feature(dataset):
    show_quantiles_feature(dataset, "Feature 1")


def test_show_quantiles_features(dataset):
    show_quantiles_features(dataset)
