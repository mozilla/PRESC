import pandas as pd
import pytest

from presc.misclassifications.misclass_rate import (
    misclass_rate_feature,
    show_misclass_rate_feature,
    show_misclass_rates_features,
    compute_quantiles,
    show_quantiles_feature,
    show_quantiles_features,
)


@pytest.fixture
def test_dataset():
    dataset = pd.DataFrame(
        {
            "Feature 1": [1., 2.5, 3.5, 5.],
            "Feature 2": [1., 6., 8., 9.],
            "Class": ["True", "True", "False", "False"]
        },
        columns=["Feature 1", "Feature 2", "Class"]
    )
    return dataset


@pytest.fixture
def test_dataset_predictions():
    dataset_predictions = ["False", "False", "False", "True"]
    return dataset_predictions


def test_misclass_rate_feature(test_dataset, test_dataset_predictions):
    bin_number = 2
    result = misclass_rate_feature(
        test_dataset,
        test_dataset_predictions,
        feature="Feature 1",
        bins=bin_number
    )
    assert len(result) == 3
    assert len(result[0]) == len(result[1]) + 1
    assert len(result[1]) == bin_number

    assert max(result[1]) <= 1.0
    assert min(result[1]) >= 0.0


def test_show_misclass_rate_feature(test_dataset, test_dataset_predictions):
    show_misclass_rate_feature(test_dataset, test_dataset_predictions, "Feature 2")


def test_show_misclass_rates_features(test_dataset, test_dataset_predictions):
    show_misclass_rates_features(test_dataset, test_dataset_predictions)


def test_compute_quantiles(test_dataset):
    selected_feature = "Feature 1"
    quantile_number = 2
    edges_bins = compute_quantiles(test_dataset, feature=selected_feature, 
                                   quantiles=quantile_number)
    assert edges_bins[0] == min(test_dataset[selected_feature])
    assert edges_bins[-1] == max(test_dataset[selected_feature])
    assert len(edges_bins) == quantile_number + 1


def test_show_quantiles_feature(test_dataset):
    show_quantiles_feature(test_dataset, "Feature 1")


def test_show_quantiles_features(test_dataset):
    show_quantiles_features(test_dataset)
