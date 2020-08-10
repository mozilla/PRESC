import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from presc.training_stability.train_test_split import explore_test_split_ratio
from presc.training_stability.training_stability_common import (
    show_averages_and_variations,
)


@pytest.fixture(
    params=[
        "accuracy",
        "balanced_accuracy",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "true_positives_fraction",
        "false_positives_fraction",
        "true_negatives_fraction",
        "false_negatives_fraction",
    ]
)
def metrics_param(request):
    return request.param


def test1_explore_test_split_ratio(metrics_param):
    dataset_binary = pd.DataFrame(
        {
            "Feature 1": [1.0, 2.5, 3.5, 2.0, 5.0],
            "Feature 2": [1.0, 6.0, 8.0, 9.0, 2.0],
            "Class": ["True", "False", "True", "False", "True"],
        },
        columns=["Feature 1", "Feature 2", "Class"],
    )
    classifier = SVC(
        kernel="linear", decision_function_shape="ovr", class_weight="balanced"
    )
    scaler = StandardScaler()

    explore_test_split_ratio(
        dataset_binary,
        classifier,
        scaler=scaler,
        metric=metrics_param,
        num_test_fractions=3,
    )


@pytest.fixture(params=["accuracy", "balanced_accuracy"])
def metrics_param_multiclass(request):
    return request.param


def test2_explore_test_split_ratio():
    dataset_multiclass = pd.DataFrame(
        {
            "Feature 1": [1.0, 2.5, 3.5, 2.0, 5.0],
            "Feature 2": [1.0, 6.0, 8.0, 9.0, 2.0],
            "Class": ["red", "blue", "green", "blue", "blue"],
        },
        columns=["Feature 1", "Feature 2", "Class"],
    )
    classifier = SVC(
        kernel="linear", decision_function_shape="ovo", class_weight="balanced"
    )
    scaler = StandardScaler()

    explore_test_split_ratio(
        dataset_multiclass,
        classifier,
        scaler=scaler,
        metric=metrics_param_multiclass,
        num_test_fractions=3,
    )


def test_show_averages_and_variations():
    x = np.array([0.2, 0.4, 0.6, 0.8])
    averages = np.array([0.65, 0.65, 0.60, 0.3])
    standard_deviations = np.array([0.05, 0.02, 0.02, 0.04])

    show_averages_and_variations(
        x,
        averages,
        standard_deviations,
        x_name="Test subset fraction",
        metric_name="accuracy",
    )
