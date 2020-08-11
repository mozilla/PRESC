import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import make_classification

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
    # Generating binary classification dataset
    dataset_binary = make_classification(n_samples=50, n_features=3, n_informative=3, n_redundant=0, n_classes=2, 
                    n_clusters_per_class=2, weights=[0.7], flip_y=0.04, class_sep=1.0, hypercube=True, 
                    shift=[0, 0.3, 1], scale=[100, 1, 5], random_state=0)

    dataset1 = pd.DataFrame(dataset_binary[0])
    dataset2 = pd.DataFrame(dataset_binary[1])
    dataset_binary = pd.concat([dataset1, dataset2], axis=1)
    dataset_binary.columns = [0, 1, 2, "class"]

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


def test2_explore_test_split_ratio(metrics_param_multiclass):
    # Generating multiple classification dataset
    dataset_multiclass = make_classification(n_samples=50, n_features=3, n_informative=3, n_redundant=0, n_classes=3, 
                    n_clusters_per_class=2, weights=[0.3, 0.4], flip_y=0.04, class_sep=1.0, hypercube=True, 
                    shift=[0, 0.3, 1], scale=[100, 1, 5], random_state=0)

    dataset1 = pd.DataFrame(dataset_multiclass[0])
    dataset2 = pd.DataFrame(dataset_multiclass[1])
    dataset_multiclass = pd.concat([dataset1, dataset2], axis=1)
    dataset_multiclass.columns = [0, 1, 2, "class"]

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
