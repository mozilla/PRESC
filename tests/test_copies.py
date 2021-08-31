import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from presc.dataset import Dataset

from presc.copies.sampling import (
    dynamical_range,
    grid_sampling,
    uniform_sampling,
    normal_sampling,
    labeling,
)
from presc.copies.evaluations import empirical_fidelity_error
from presc.copies.copying import ClassifierCopy
from presc.copies.examples import multiclass_gaussians


def test_dynamical_range():
    data = {
        "age": [33, 21, 42, 80],
        "weight": [70.5, 80.3, 55.8, 65.1],
        "height": [165, 187, 159, 170],
    }
    df = pd.DataFrame(data, columns=["age", "weight", "height"])
    range_dict = dynamical_range(df)
    expected_range_dict = {
        "age": {"min": 21, "max": 80, "mean": 44, "sigma": 25.5},
        "weight": {"min": 55.8, "max": 80.3, "mean": 67.9, "sigma": 10.2},
        "height": {"min": 159, "max": 187, "mean": 170, "sigma": 12},
    }
    assert range_dict.keys() == expected_range_dict.keys()
    for key in range_dict:
        assert range_dict[key].keys() == expected_range_dict[key].keys()
        for descriptor in range_dict[key]:
            np.testing.assert_approx_equal(
                range_dict[key][descriptor],
                expected_range_dict[key][descriptor],
                significant=3,
            )


def test_grid_sampling():
    feature_parameters = {
        "feat_1": {"min": 0, "max": 2},
        "feat_2": {"min": 20, "max": 40},
    }
    df_test_1 = grid_sampling(
        nsamples=342, random_state=2, feature_parameters=feature_parameters
    )
    df_test_2 = grid_sampling(
        nsamples=342, random_state=2, feature_parameters=feature_parameters
    )
    df_test_3 = grid_sampling(
        nsamples=342, random_state=6, feature_parameters=feature_parameters
    )

    assert len(df_test_1) == 324
    assert len(df_test_1["feat_1"].unique()) == len(
        df_test_1["feat_2"].unique()
    )  # is True
    assert df_test_1.equals(df_test_2) is True
    assert df_test_1.equals(df_test_3) is True
    assert df_test_1["feat_1"].max() <= feature_parameters["feat_1"]["max"]
    assert df_test_1["feat_2"].min() >= feature_parameters["feat_2"]["min"]


def test_uniform_sampling():
    feature_parameters = {
        "feat_1": {"min": 0, "max": 2},
        "feat_2": {"min": 20, "max": 40},
    }
    df_test_1 = uniform_sampling(
        nsamples=342, random_state=2, feature_parameters=feature_parameters
    )
    df_test_2 = uniform_sampling(
        nsamples=342, random_state=2, feature_parameters=feature_parameters
    )
    df_test_3 = uniform_sampling(
        nsamples=342, random_state=6, feature_parameters=feature_parameters
    )

    assert len(df_test_1) == 342
    assert df_test_1.equals(df_test_2) is True
    assert df_test_1.equals(df_test_3) is False
    assert df_test_1["feat_1"].max() <= feature_parameters["feat_1"]["max"]
    assert df_test_1["feat_2"].min() >= feature_parameters["feat_2"]["min"]


def test_normal_sampling():
    feature_parameters = {
        "feat_1": {"mean": 0, "sigma": 2},
        "feat_2": {"mean": 20, "sigma": 40},
    }
    df_test_1 = normal_sampling(
        nsamples=342, random_state=2, feature_parameters=feature_parameters
    )
    df_test_2 = normal_sampling(
        nsamples=342, random_state=2, feature_parameters=feature_parameters
    )
    df_test_3 = normal_sampling(
        nsamples=342, random_state=6, feature_parameters=feature_parameters
    )

    assert len(df_test_1) == 342
    assert df_test_1.equals(df_test_2) is True
    assert df_test_1.equals(df_test_3) is False
    np.testing.assert_almost_equal(df_test_1["feat_1"].mean(), 0, decimal=0)
    np.testing.assert_almost_equal(df_test_1["feat_2"].mean(), 20, decimal=0)
    np.testing.assert_almost_equal(df_test_1["feat_1"].std(), 2, decimal=0)
    np.testing.assert_almost_equal(df_test_1["feat_2"].std(), 40, decimal=0)


def test_labeling():
    train_dataset = pd.DataFrame(
        np.random.rand(4, 3) * 80 + np.tile(np.array([10, 40, 130]), (4, 1))
    )
    original_classifier = DummyClassifier(strategy="constant", constant="dummy_class")
    original_classifier.fit(train_dataset, np.array(["a", "b", "dummy_class", "a"]))

    unlabeled_dataset = pd.DataFrame(
        np.random.rand(4, 3) * 80 + np.tile(np.array([10, 40, 130]), (4, 1))
    )
    label_col = "potato"
    df_labeled = labeling(unlabeled_dataset, original_classifier, label_col=label_col)

    assert isinstance(df_labeled, Dataset) is True
    assert len(df_labeled.df["potato"]) == 4
    assert df_labeled.df["potato"].unique() == "dummy_class"


def test_empirical_fidelity_error():
    y_pred_original = [1, 0, 1, 0]
    y_pred_copy1 = [1, 1, 0, 0]
    y_pred_copy2 = [1, 0, 0, 0]
    efe1 = empirical_fidelity_error(y_pred_original, y_pred_copy1)
    efe2 = empirical_fidelity_error(y_pred_original, y_pred_copy2)
    assert efe1 == 0.5
    assert efe2 == 0.25


def test_ClassifierCopy_copy_classifier():
    # Original classifier
    train_data = pd.DataFrame(
        {"x": [0, 1, 0, 2, 1], "y": [1, 0, 2, 0, 1], "label": [0, 0, 1, 1, 1]},
        columns=["x", "y", "label"],
    )
    original_classifier = SVC(kernel="linear", random_state=42)
    original_classifier.fit(train_data[["x", "y"]], train_data["label"])

    # Copy classifier
    feature_parameters = {"x": {"min": 0, "max": 2}, "y": {"min": 0, "max": 2}}
    classifier_copy = DecisionTreeClassifier(max_depth=2, random_state=42)
    copy_grid = ClassifierCopy(
        original_classifier,
        classifier_copy,
        grid_sampling,
        nsamples=900,
        label_col="label",
        feature_parameters=feature_parameters,
    )
    train_data_copy = copy_grid.copy_classifier(get_training_data=True)

    assert isinstance(train_data_copy, Dataset) is True
    assert len(train_data_copy.df["label"]) == 900
    assert list(train_data_copy.df["label"].unique()) == [0, 1]

    # Evaluate classifier
    test_data = pd.DataFrame(
        {"x": [-1, 0, 1, 2], "y": [-1, 0, 1, 2]}, columns=["x", "y"]
    )
    efe = copy_grid.compute_fidelity_error(test_data=test_data)

    assert efe == 0


def test_multiclass_gaussians():
    nsamples = 1500
    nclasses = 15
    nfeatures = 30
    df_2feat_2class = multiclass_gaussians(
        nsamples=nsamples,
        nfeatures=nfeatures,
        nclasses=nclasses,
        center_low=100,
        center_high=120,
        scale_low=2,
        scale_high=3,
    ).df

    assert len(df_2feat_2class) == nsamples - (nsamples % nclasses)
    assert len(df_2feat_2class["class"].unique()) == nclasses

    for class_number in range(nclasses):
        class_df = df_2feat_2class[df_2feat_2class["class"] == class_number]
        for feature in range(nfeatures):
            if class_number == 0:
                assert class_df[feature].mean() >= -1
                assert class_df[feature].mean() <= 1
            assert class_df[feature].std() >= 1.5
            assert class_df[feature].std() <= 4.0
