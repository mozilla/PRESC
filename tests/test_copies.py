import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytest
import time

from queue import Queue
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from presc.dataset import Dataset

from presc.copies.sampling import (
    dynamical_range,
    reduce_feature_space,
    find_categories,
    build_equal_category_dict,
    mixed_data_features,
    grid_sampling,
    uniform_sampling,
    normal_sampling,
    categorical_sampling,
    mixed_data_sampling,
    labeling,
    sampling_balancer,
)
from presc.copies.evaluations import (
    empirical_fidelity_error,
    replacement_capability,
    summary_metrics,
    multivariable_density_comparison,
    keep_top_classes,
)
from presc.copies.continuous import (
    SyntheticDataStreamer,
    ContinuousCopy,
    check_partial_fit,
)
from presc.copies.copying import ClassifierCopy
from presc.copies.examples import multiclass_gaussians


@pytest.fixture
def example_presc_datasets():
    data_points = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5],
    ]
    labels1 = [["a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c"]]
    labels2 = [["b", "b", "b", "b", "b", "a", "a", "a", "a", "c", "c"]]

    test_dataframe_1 = pd.DataFrame(
        np.array(data_points + labels1).T, columns=["feature1", "feature2", "letter"]
    )
    test_dataframe_1[["feature1", "feature2"]] = test_dataframe_1[
        ["feature1", "feature2"]
    ].astype(float)
    test_dataframe_2 = pd.DataFrame(
        np.array(data_points + labels2).T, columns=["feature1", "feature2", "letter"]
    )
    test_dataframe_2[["feature1", "feature2"]] = test_dataframe_2[
        ["feature1", "feature2"]
    ].astype(float)

    test_presc_dataset_1 = Dataset(test_dataframe_1, label_col="letter")
    test_presc_dataset_2 = Dataset(test_dataframe_2, label_col="letter")
    test_presc_datasets = [test_presc_dataset_1, test_presc_dataset_2]
    return test_presc_datasets


@pytest.fixture
def train_data():
    train_data = pd.DataFrame(
        {"x": [0, 1, 0, 2, 1], "y": [1, 0, 2, 0, 1], "label": [0, 0, 1, 1, 1]},
        columns=["x", "y", "label"],
    )
    return train_data


@pytest.fixture
def trained_original_classifier(train_data):
    original_classifier = SVC(kernel="linear", random_state=42)
    original_classifier.fit(train_data[["x", "y"]], train_data["label"])
    return original_classifier


@pytest.fixture
def instantiated_classifier_copies(trained_original_classifier):
    # Feature space description
    feature_parameters = {"x": {"min": 0, "max": 2}, "y": {"min": 0, "max": 2}}

    # Instantiate ClassifierCopy with two sampling options and parameters
    classifier_copy_grid = DecisionTreeClassifier(max_depth=2, random_state=42)
    copy_grid = ClassifierCopy(
        trained_original_classifier,
        classifier_copy_grid,
        grid_sampling,
        nsamples=900,
        label_col="label",
        feature_parameters=feature_parameters,
    )

    classifier_copy_uniform = DecisionTreeClassifier()
    copy_uniform = ClassifierCopy(
        trained_original_classifier,
        classifier_copy_uniform,
        uniform_sampling,
        nsamples=10,
        label_col="label",
        feature_parameters=feature_parameters,
    )
    return {"grid_copy": copy_grid, "uniform_copy": copy_uniform}


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


def test_reduce_feature_space():
    feature_parameters = {
        "feature1": {"min": 0, "max": 1000, "mean": 50, "sigma": None},
        "feature2": {"categories": {"red": 0.4, "blue": 0.6}},
        "feature3": {"min": -100, "max": 20, "mean": -0.001, "sigma": 0.0005},
        "feature4": {"min": 10, "max": 20},
    }
    reduced_feature_space_2 = reduce_feature_space(feature_parameters, sigmas=2)
    expected_reduced_feature_space_2 = {
        "feature1": {"min": 0, "max": 1000, "mean": 50, "sigma": None},
        "feature2": {"categories": {"red": 0.4, "blue": 0.6}},
        "feature3": {"min": -0.002, "max": 0, "mean": -0.001, "sigma": 0.0005},
        "feature4": {"min": 10, "max": 20},
    }
    assert reduced_feature_space_2 == expected_reduced_feature_space_2


def test_find_categories():
    data = {
        "color": ["red", "blue", "red", "blue", "blue"],
        "height": [70.5, 80.3, 55.8, 65.1, 65.4],
        "siblings": [1, 1, 1, 2, np.nan],
    }
    df = pd.DataFrame(data, columns=["color", "height", "siblings"])
    df[["color", "siblings"]] = df[["color", "siblings"]].astype("category")
    category_dict = find_categories(df, add_nans=True)
    expected_category_dict = {
        "color": {"categories": {"red": 0.4, "blue": 0.6}},
        "siblings": {"categories": {1.0: 0.6, 2.0: 0.2, "NaNs": 0.2}},
    }
    assert category_dict.keys() == expected_category_dict.keys()
    for key in category_dict:
        assert (
            category_dict[key]["categories"].keys()
            == expected_category_dict[key]["categories"].keys()
        )
        for descriptor in category_dict[key]["categories"]:
            np.testing.assert_approx_equal(
                category_dict[key]["categories"][descriptor],
                expected_category_dict[key]["categories"][descriptor],
                significant=3,
            )


def test_build_equal_category_dict():
    category_lists = {"color": ["red", "blue"], "siblings": [0, 1, 2, 5, 9]}
    category_dict = build_equal_category_dict(category_lists)
    expected_category_dict = {
        "color": {"categories": {"red": 0.5, "blue": 0.5}},
        "siblings": {"categories": {0: 0.2, 1: 0.2, 2: 0.2, 5: 0.2, 9: 0.2}},
    }
    assert category_dict.keys() == expected_category_dict.keys()
    for key in category_dict:
        assert (
            category_dict[key]["categories"].keys()
            == expected_category_dict[key]["categories"].keys()
        )
        for descriptor in category_dict[key]["categories"]:
            np.testing.assert_approx_equal(
                category_dict[key]["categories"][descriptor],
                expected_category_dict[key]["categories"][descriptor],
                significant=3,
            )


def test_mixed_data_features():
    data = {
        "color": ["red", "blue", "red", "blue", "blue"],
        "height": [70.5, 80.3, 55.8, 65.1, 65.4],
        "siblings": [1, 1, 1, 2, np.nan],
    }
    df = pd.DataFrame(data, columns=["color", "height", "siblings"])
    df[["color", "siblings"]] = df[["color", "siblings"]].astype("category")
    feature_dict = mixed_data_features(df, add_nans=True)
    expected_feature_dict = {
        "color": {"categories": {"red": 0.4, "blue": 0.6}},
        "height": {"min": 55.8, "max": 80.3, "mean": 67.42, "sigma": 8.94242696},
        "siblings": {"categories": {1.0: 0.6, 2.0: 0.2, "NaNs": 0.2}},
    }
    assert feature_dict.keys() == expected_feature_dict.keys()
    for key in feature_dict:
        if "categories" in key:
            assert (
                feature_dict[key]["categories"].keys()
                == expected_feature_dict[key]["categories"].keys()
            )
            for descriptor in feature_dict[key]["categories"]:
                np.testing.assert_approx_equal(
                    feature_dict[key]["categories"][descriptor],
                    expected_feature_dict[key]["categories"][descriptor],
                    significant=3,
                )
        for descriptor in feature_dict[key]:
            if descriptor != "categories":
                np.testing.assert_approx_equal(
                    feature_dict[key][descriptor],
                    expected_feature_dict[key][descriptor],
                    significant=3,
                )


def test_grid_sampling():
    feature_parameters = {
        "feat_1": {"min": 0, "max": 2},
        "feat_2": {"min": 20, "max": 40},
    }
    df_test_1 = grid_sampling(feature_parameters, nsamples=342, random_state=2)
    df_test_2 = grid_sampling(feature_parameters, nsamples=342, random_state=2)
    df_test_3 = grid_sampling(feature_parameters, nsamples=342, random_state=6)

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
    df_test_1 = uniform_sampling(feature_parameters, nsamples=342, random_state=2)
    df_test_2 = uniform_sampling(feature_parameters, nsamples=342, random_state=2)
    df_test_3 = uniform_sampling(feature_parameters, nsamples=342, random_state=6)

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
    df_test_1 = normal_sampling(feature_parameters, nsamples=342, random_state=2)
    df_test_2 = normal_sampling(feature_parameters, nsamples=342, random_state=2)
    df_test_3 = normal_sampling(feature_parameters, nsamples=342, random_state=6)

    assert len(df_test_1) == 342
    assert df_test_1.equals(df_test_2) is True
    assert df_test_1.equals(df_test_3) is False
    np.testing.assert_almost_equal(df_test_1["feat_1"].mean(), 0, decimal=0)
    np.testing.assert_almost_equal(df_test_1["feat_2"].mean(), 20, decimal=0)
    np.testing.assert_almost_equal(df_test_1["feat_1"].std(), 2, decimal=0)
    np.testing.assert_almost_equal(df_test_1["feat_2"].std(), 40, decimal=0)


def test_categorical_sampling():
    feature_parameters = {
        "feat_1": {"categories": {"red": 0.5, "blue": 0.5}},
        "feat_2": {"categories": {1: 0.3, 3: 0.6, 69: 0.1}},
    }
    data_generated = categorical_sampling(
        feature_parameters, nsamples=10000, random_state=2
    )
    for key in feature_parameters:
        assert key in data_generated.columns
        for category in feature_parameters[key]["categories"]:
            obtained_fraction = (
                data_generated[key].value_counts().loc[[category]].iloc[0] / 10000
            )
            expected_probability = feature_parameters[key]["categories"][category]
            assert obtained_fraction == pytest.approx(expected_probability, 0.05)
    assert len(data_generated) == 10000


def test_mixed_data_sampling():
    feature_parameters = {
        "feat_1": {"categories": {1: 0.3, 3: 0.6, 69: 0.1}},
        "feat_2": {"mean": 20, "sigma": 40},
    }
    data_generated = mixed_data_sampling(
        feature_parameters,
        normal_sampling,
        nsamples=10000,
        random_state=2,
    )
    assert list(feature_parameters.keys()) == list(data_generated.columns)
    for key in feature_parameters:
        if "categories" in feature_parameters[key]:
            for category in feature_parameters[key]["categories"]:
                obtained_fraction = (
                    data_generated[key].value_counts().loc[[category]].iloc[0] / 10000
                )
                expected_probability = feature_parameters[key]["categories"][category]
                assert obtained_fraction == pytest.approx(expected_probability, 0.05)
        else:
            np.testing.assert_almost_equal(
                data_generated[key].mean(), feature_parameters[key]["mean"], decimal=0
            )
            np.testing.assert_almost_equal(
                data_generated[key].std(), feature_parameters[key]["sigma"], decimal=0
            )
    assert len(data_generated) == 10000


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


def test_sampling_balancer():
    # Build an 'original' model that can label the samples
    # This dummy model will only yield 10% of the predictions as class 0
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    original_classifier = DummyClassifier(strategy="stratified")
    original_classifier.fit(x, y)

    feature_parameters = {"feature": {"mean": 0, "sigma": 10}}
    generated_data = sampling_balancer(
        feature_parameters,
        normal_sampling,
        original_classifier,
        nsamples=100,
        max_iter=12,
        nbatch=100,
        label_col="class",
        random_state=42,
    )

    assert (
        generated_data["class"].value_counts()[0]
        == generated_data["class"].value_counts()[1]
    )


def test_empirical_fidelity_error():
    y_pred_original = [1, 0, 1, 0]
    y_pred_copy1 = [1, 1, 0, 0]
    y_pred_copy2 = [1, 0, 0, 0]
    efe1 = empirical_fidelity_error(y_pred_original, y_pred_copy1)
    efe2 = empirical_fidelity_error(y_pred_original, y_pred_copy2)
    assert efe1 == 0.5
    assert efe2 == 0.25


def test_replacement_capability():
    y_true = [1, 0, 1, 0, 1]
    y_pred_original = [1, 0, 1, 1, 0]  # 3 right predictions
    y_pred_copy1 = [1, 1, 0, 0, 0]  # 2 right predictions
    y_pred_copy2 = [1, 0, 1, 0, 0]  # 4 right predictions
    rc1 = replacement_capability(y_true, y_pred_original, y_pred_copy1)
    rc2 = replacement_capability(y_true, y_pred_original, y_pred_copy2)

    np.testing.assert_almost_equal(rc1, 2.0 / 3.0, decimal=14)
    np.testing.assert_almost_equal(rc2, 4.0 / 3.0, decimal=14)


def test_summary_metrics():
    random_seed = 42
    np.random.seed(random_seed)

    # Original data
    train_data = pd.DataFrame(
        {"x": [0, 1, 0, 2, 1], "y": [1, 0, 2, 0, 1], "label": [0, 0, 1, 1, 1]},
        columns=["x", "y", "label"],
    )
    test_data = Dataset(
        pd.DataFrame(
            {"x": [2, 0, 0, 1, 2], "y": [1, 0, 2, 0, 2], "label": [0, 0, 1, 0, 1]},
            columns=["x", "y", "label"],
        ),
        label_col="label",
    )

    # Original classifier
    original_classifier = SVC(kernel="linear", random_state=random_seed)
    original_classifier.fit(train_data[["x", "y"]], train_data["label"])

    # Copy classifier
    feature_parameters = {"x": {"min": 0, "max": 2}, "y": {"min": 0, "max": 2}}
    classifier_copy = DecisionTreeClassifier(max_depth=2, random_state=random_seed)
    copy_grid = ClassifierCopy(
        original_classifier,
        classifier_copy,
        grid_sampling,
        nsamples=20,
        label_col="label",
        feature_parameters=feature_parameters,
    )
    copy_grid.copy_classifier()

    # Generated data
    synthetic_test_data = copy_grid.generate_synthetic_data(
        nsamples=100,
        random_state=random_seed,
    )

    metrics = summary_metrics(
        original_model=original_classifier,
        copy_model=copy_grid,
        test_data=test_data,
        synthetic_data=synthetic_test_data,
        show_results=True,
    )

    expected_results = {
        "Original Model Accuracy (test)": 0.6,
        "Copy Model Accuracy (test)": 0.8,
        "Empirical Fidelity Error (synthetic)": 0.1,
        "Empirical Fidelity Error (test)": 0.2,
        "Replacement Capability (synthetic)": 0.9,
        "Replacement Capability (test)": 1.33333333,
    }

    metric_names = metrics.keys()
    for name in metric_names:
        np.testing.assert_almost_equal(metrics[name], expected_results[name], decimal=6)


def test_multivariable_density_comparison(example_presc_datasets):
    num_figures_before = plt.gcf().number
    multivariable_density_comparison(
        [example_presc_datasets[0].df[:-2], example_presc_datasets[1].df[:-2]],
        feature1="feature1",
        feature2="feature2",
        label_col="letter",
        titles=["Classifier 1", "Classifier 2"],
    )
    num_figures_after = plt.gcf().number
    # Checks that the figure was plotted
    assert num_figures_after == num_figures_before + 1


def test_keep_top_classes(example_presc_datasets):
    min_num_samples = 3
    classes_to_keep = ["b", "c"]

    dataset_majority_classes = keep_top_classes(
        example_presc_datasets[0], min_num_samples=min_num_samples
    )
    dataset_specified_classes = keep_top_classes(
        example_presc_datasets[0],
        min_num_samples=min_num_samples,
        classes_to_keep=classes_to_keep,
    )

    assert dataset_majority_classes.labels.isin(["a", "b"]).all()
    assert dataset_specified_classes.labels.isin(["b", "c"]).all()


def test_SyntheticDataStreamer(
    trained_original_classifier, instantiated_classifier_copies
):
    # Define queue
    data_stream = Queue(maxsize=4)

    # Instantiate and start data streamer
    classifier_copy = instantiated_classifier_copies["uniform_copy"]
    data_streamer = SyntheticDataStreamer(classifier_copy, data_stream, verbose=True)
    data_streamer.daemon = True
    data_streamer.start()

    # Assert data_streamer is running
    assert data_streamer.is_alive()

    # Assert that the queue is full and it has the proper length
    time.sleep(1)
    print(data_stream.queue)
    assert data_stream.full()
    assert data_stream.qsize() == 4

    # Assert that elements in queue are the expected datasets
    data_batch = data_stream.get()
    assert isinstance(data_batch, Dataset)
    assert len(data_batch.df) == 10

    # Stop data streamer thread
    data_streamer.stop()
    _ = data_stream.get()

    # Assert that data streamer really stopped
    time.sleep(1)
    assert not data_streamer.is_alive()


def test_ContinuousCopy(
    train_data, trained_original_classifier, instantiated_classifier_copies
):
    # Define queue
    data_stream = Queue(maxsize=3)

    # Add some data blocks to queue manually
    data_block_1 = Dataset(train_data[0:2], label_col="label")
    data_block_2 = Dataset(train_data[2:3], label_col="label")
    data_block_3 = Dataset(train_data[3:5], label_col="label")
    data_stream.put(data_block_1)
    data_stream.put(data_block_2)
    data_stream.put(data_block_3)

    # Instantiate the copy pipepline
    sdg_normal_classifier = Pipeline(
        [("scaler", StandardScaler()), ("sdg_classifier", SGDClassifier())]
    )

    # Instantiate the copier class
    feature_parameters = {"x": {"min": 0, "max": 2}, "y": {"min": 0, "max": 2}}
    sdg_normal_copy = ClassifierCopy(
        trained_original_classifier,
        sdg_normal_classifier,
        uniform_sampling,
        random_state=42,
        feature_parameters=feature_parameters,
        label_col="label",
    )

    # Instantiate and start continous copy
    fit_kwargs = {"scaler": {}, "sdg_classifier": {"classes": [0, 1]}}
    online_copy = ContinuousCopy(
        sdg_normal_copy,
        data_stream,
        fit_kwargs=fit_kwargs,
        verbose=True,
        test_data=data_block_2,
    )
    online_copy.daemon = True
    online_copy.start()

    # Assert online copy is running
    assert online_copy.is_alive()

    # Assert that the queue is empty and it has the proper length
    time.sleep(1)
    assert data_stream.empty()
    assert data_stream.qsize() == 0

    # Assert number of iterations and samples
    assert online_copy.iterations == 3
    assert online_copy.n_samples == 5

    # Stop online copy thread
    online_copy.stop()
    data_stream.put(0)

    # Assert that online copy really stopped
    time.sleep(1)
    assert not online_copy.is_alive()


def test_check_partial_fit():
    # Instantiate classifier with and without incremental training
    classifier_without = DummyClassifier()
    classifier_with = SGDClassifier()
    # Instantiate pipeline with and without incremental training
    pipeline_without = Pipeline(
        [("scaler", StandardScaler()), ("tree_classifier", DummyClassifier())]
    )
    pipeline_with = Pipeline(
        [("scaler", StandardScaler()), ("sdg_classifier", SGDClassifier())]
    )

    # Check expected return
    assert not check_partial_fit(classifier_without)
    assert check_partial_fit(classifier_with)
    assert not check_partial_fit(pipeline_without)
    assert check_partial_fit(pipeline_with)


def test_ClassifierCopy_copy_classifier(
    trained_original_classifier, instantiated_classifier_copies
):
    # Copy classifier
    copy_grid = instantiated_classifier_copies["grid_copy"]
    train_data_copy = copy_grid.copy_classifier(get_training_data=True, nsamples=900)

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
