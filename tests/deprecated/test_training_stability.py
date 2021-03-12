import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

from presc.deprecated.training_stability.train_test_split import (
    explore_test_split_ratio,
)
from presc.deprecated.training_stability.cross_validation import (
    explore_cross_validation_kfolds,
)
from presc.deprecated.training_stability.training_stability_common import (
    show_averages_and_variations,
)


@pytest.fixture
def dataset_binary():
    """Generate binary classification dataset."""
    dataset = make_classification(
        n_samples=50,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.7],
        flip_y=0.04,
        class_sep=1.0,
        hypercube=True,
        shift=[0, 0.3, 1],
        scale=[100, 1, 5],
        random_state=0,
    )

    dataset1 = pd.DataFrame(dataset[0])
    dataset2 = pd.DataFrame(dataset[1])
    dataset = pd.concat([dataset1, dataset2], axis=1)
    dataset.columns = [0, 1, 2, "class"]
    return dataset


@pytest.fixture
def dataset_multiclass():
    """Generate multiple classification dataset."""
    dataset = make_classification(
        n_samples=50,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=2,
        weights=[0.3, 0.4],
        flip_y=0.04,
        class_sep=1.0,
        hypercube=True,
        shift=[0, 0.3, 1],
        scale=[100, 1, 5],
        random_state=0,
    )

    dataset1 = pd.DataFrame(dataset[0])
    dataset2 = pd.DataFrame(dataset[1])
    dataset = pd.concat([dataset1, dataset2], axis=1)
    dataset.columns = [0, 1, 2, "class"]
    return dataset


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


def test_explore_test_split_ratio_binary(dataset_binary, metrics_param):

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


def test_explore_test_split_ratio_multiclass(
    dataset_multiclass, metrics_param_multiclass
):

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


def test_explore_cross_validation_kfolds_binary(dataset_binary):

    # Define parameters
    scaler = StandardScaler()
    classifier = SVC(kernel="linear", C=1, random_state=0)
    pipeline = Pipeline([("scaler", scaler), ("classifier", classifier)])
    metrics = "accuracy"

    kfolds_list = [2, 5, 20]
    explore_cross_validation_kfolds(
        dataset_binary,
        pipeline,
        metrics=metrics,
        kfolds_list=kfolds_list,
        repetitions=1,
        minimum_kfolds=False,
    )

    explore_cross_validation_kfolds(
        dataset_binary,
        pipeline,
        metrics=metrics,
        kfolds_list=kfolds_list,
        repetitions=5,
        minimum_kfolds=True,
    )

    kfolds_list = [2, 5, 10]
    explore_cross_validation_kfolds(
        dataset_binary,
        pipeline,
        metrics=metrics,
        kfolds_list=kfolds_list,
        repetitions=5,
        minimum_kfolds=False,
    )


def test_explore_cross_validation_kfolds_multiclass(dataset_multiclass):

    # Define parameters
    scaler = StandardScaler()
    classifier = SVC(kernel="linear", C=1, random_state=0)
    pipeline = Pipeline([("scaler", scaler), ("classifier", classifier)])
    metrics = "accuracy"

    kfolds_list = [2, 5, 20]
    explore_cross_validation_kfolds(
        dataset_multiclass,
        pipeline,
        metrics=metrics,
        kfolds_list=kfolds_list,
        repetitions=1,
        minimum_kfolds=False,
    )

    explore_cross_validation_kfolds(
        dataset_multiclass,
        pipeline,
        metrics=metrics,
        kfolds_list=kfolds_list,
        repetitions=5,
        minimum_kfolds=True,
    )

    kfolds_list = [2, 5, 10]
    explore_cross_validation_kfolds(
        dataset_multiclass,
        pipeline,
        metrics=metrics,
        kfolds_list=kfolds_list,
        repetitions=5,
        minimum_kfolds=False,
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
