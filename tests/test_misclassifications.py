import pandas as pd
import pytest

from presc.misclassifications.misclassification_visuals import predictions_to_class
from presc.misclassifications.misclass_rate import (
    misclass_rate_feature,
    show_misclass_rate_feature,
    show_misclass_rates_features,
    compute_tiles,
    show_tiles_feature,
    show_tiles_features,
)


def test_predictions_to_class():
    X_test = pd.DataFrame([10, 20, 30, 40])
    y_test = pd.DataFrame([True, False, True, False])
    y_predicted = pd.DataFrame([True, False, False, True])
    options = ["hits-fails", "which-hit", "which-fail"]

    # "hits-fails" yields: "> Prediction hit" + "> Prediction fail".
    # "which-hit" yields: original classes (as strings) + "> Prediction fail".
    # "which-fail" yields: "> Prediction hit" + original classes (as strings).

    for analysis in options:
        new_dataset = predictions_to_class(
            X_test, y_test, y_predicted, new_classes=analysis
        )

        assert type(new_dataset) is pd.DataFrame
        assert len(new_dataset) == 4
        assert len(new_dataset.columns) == 2

        if analysis == "hits-fails":
            assert "hit" in new_dataset.iloc[0, 1]
            assert "hit" in new_dataset.iloc[1, 1]
            assert "fail" in new_dataset.iloc[2, 1]
            assert "fail" in new_dataset.iloc[3, 1]

        elif analysis == "which-hit":
            assert "True" in new_dataset.iloc[0, 1]
            assert "False" in new_dataset.iloc[1, 1]
            assert "fail" in new_dataset.iloc[2, 1]
            assert "fail" in new_dataset.iloc[3, 1]

        elif analysis == "which-fail":
            assert "hit" in new_dataset.iloc[0, 1]
            assert "hit" in new_dataset.iloc[1, 1]
            assert "True" in new_dataset.iloc[2, 1]
            assert "False" in new_dataset.iloc[3, 1]


@pytest.fixture
def dataset_predictions2class():
    dataset = pd.DataFrame(
        {
            "Feature 1": [1, 2.5, 3.5, 5],
            "Feature 2": [1, 6, 8, 9],
            "Miss & Class": ["True", "False", "> Prediction hit", "True"],
        },
        columns=["Feature 1", "Feature 2", "Miss & Class"],
    )
    return dataset


@pytest.fixture
def dataset_predictions2class_misclass():
    dataset_misclass = pd.DataFrame(
        {
            "Feature 1": [1, 2.5, 5],
            "Feature 2": [1, 6, 9],
            "Miss & Class": ["True", "False", "True"],
        },
        columns=["Feature 1", "Feature 2", "Miss & Class"],
    )
    return dataset_misclass


def test_misclass_rate_feature(
    dataset_predictions2class, dataset_predictions2class_misclass
):
    bin_number = 2
    result = misclass_rate_feature(
        dataset_predictions2class,
        dataset_predictions2class_misclass,
        feature="Feature 1",
        bins=bin_number,
    )
    assert len(result) == 3
    assert len(result[0]) == len(result[1]) + 1
    assert len(result[1]) == bin_number

    assert max(result[1]) <= 1.0
    assert min(result[1]) >= 0.0


def test_show_misclass_rate_feature(
    dataset_predictions2class, dataset_predictions2class_misclass
):
    show_misclass_rate_feature(
        dataset_predictions2class, dataset_predictions2class_misclass, "Feature 2"
    )


def test_show_misclass_rates_features(
    dataset_predictions2class, dataset_predictions2class_misclass
):
    show_misclass_rates_features(
        dataset_predictions2class, dataset_predictions2class_misclass
    )


@pytest.fixture
def dataset2_predictions2class():
    dataset = pd.DataFrame(
        {
            "Feature 1": [1, 2.5, 3.5, 5],
            "Feature 2": [1, 6, 8, 9],
            "Miss & Class": ["True", "False", "> Prediction hit", "True"],
        },
        columns=["Feature 1", "Feature 2", "Miss & Class"],
    )
    return dataset


def test_compute_tiles(dataset2_predictions2class):
    selected_feature = "Feature 1"
    tile_number = 2

    edges_bins = compute_tiles(
        dataset2_predictions2class, feature=selected_feature, tiles=tile_number
    )

    assert edges_bins[0] == min(dataset2_predictions2class[selected_feature])
    assert edges_bins[-1] == max(dataset2_predictions2class[selected_feature])
    assert len(edges_bins) == tile_number + 1


def test_show_tiles_feature(dataset2_predictions2class):
    show_tiles_feature(dataset2_predictions2class, "Feature 1")


def test_show_tiles_features(dataset2_predictions2class):
    show_tiles_features(dataset2_predictions2class)
