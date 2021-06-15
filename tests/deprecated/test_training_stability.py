import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, confusion_matrix

from presc.deprecated.training_stability.cross_validation import (
    explore_cross_validation_kfolds,
)
from presc.dataset import Dataset


# --------------------
# This is no longer needed.
# It has been replaced by presc.evaluations.train_test_splits.compute_train_test_splits
# Keeping here so as not to break the tests.
#
# The logic of compute_train_test_splits is slightly different from the previous
# explore_test_split_ratio. However, since the tests only check that it runs
# without error, this just calls the new version without reconciling the logic.
# --------------------
def explore_test_split_ratio(
    dataset,
    classifier,
    scaler=None,
    metric="accuracy",
    num_test_fractions=10,
    random_tries=10,
):
    from presc.evaluations.train_test_splits import compute_train_test_splits as ctts

    result = ctts(
        Dataset(dataset, dataset.columns[-1]),
        classifier,
        get_scoring(metric),
        split_size_increment=1 / num_test_fractions,
        num_replicates=random_tries,
    )
    result.display_result("abc")


# --------------------
# This is no longer needed.
# Keeping here so as not to break the tests.
# --------------------
def get_scoring(metric):
    if metric == "true_negatives":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[0, 0])
    if metric == "false_positives":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[0, 1])
    if metric == "false_negatives":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[1, 0])
    if metric == "true_positives":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[1, 1])
    if metric == "true_negatives_fraction":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[0, 0] / len(yp))
    if metric == "false_positives_fraction":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[0, 1] / len(yp))
    if metric == "false_negatives_fraction":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[1, 0] / len(yp))
    if metric == "true_positives_fraction":
        return make_scorer(lambda y, yp: confusion_matrix(y, yp)[1, 1] / len(yp))
    return metric


# --------------------
# Not sure we need this - we already have a generated dataset fixture.
# --------------------
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


# --------------------
# The generated dataset fixture we already have is only binary class - might be
# good to have multiclass as well.
# --------------------
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


# --------------------
# This could be a part of testing for presc.evaluations.train_test_splits
# Currently this only checks the computation runs without error but doesn't
# validate the outputs. It runs over a large variety of metrics (probably don't
# need all of these).
# --------------------
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


# --------------------
# This could be a part of testing for presc.evaluations.train_test_splits
# Currently this only checks the computation runs without error but doesn't
# validate the outputs. It runs over a large variety of metrics (probably don't
# need all of these).
# --------------------
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


# --------------------
# This tests
# presc.deprecated.training_stability.cross_validation.explore_cross_validation_kfolds
# This has not yet been ported to the new evaluation framework (#273)
# --------------------
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


# --------------------
# This tests
# presc.deprecated.training_stability.cross_validation.explore_cross_validation_kfolds
# This has not yet been ported to the new evaluation framework (#273)
# --------------------
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


# --------------------
# This just checks that the plotting runs without error.
# This has been integrated into test_explore_test_split_ratio_* above
# by calling `result.display_result()`
# --------------------
# def test_show_averages_and_variations():
#     x = np.array([0.2, 0.4, 0.6, 0.8])
#     averages = np.array([0.65, 0.65, 0.60, 0.3])
#     standard_deviations = np.array([0.05, 0.02, 0.02, 0.04])

#     show_averages_and_variations(
#         x,
#         averages,
#         standard_deviations,
#         x_name="Test subset fraction",
#         metric_name="accuracy",
#     )
