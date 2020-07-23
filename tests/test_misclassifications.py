from presc.misclassifications.misclassification_visuals import *
from presc.misclassifications.misclass_rate import *


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


def test_misclass_rate_feature():
    dataset = pd.DataFrame(
        {
            "Feature 1": [1, 2.5, 3.5, 5],
            "Feature 2": [1, 6, 8, 9],
            "Miss & Class": ["True", "False", "> Prediction hit", "True"],
        },
        columns=["Feature 1", "Feature 2", "Miss & Class"],
    )
    dataset_misclass = pd.DataFrame(
        {
            "Feature 1": [1, 2.5, 5],
            "Feature 2": [1, 6, 9],
            "Miss & Class": ["True", "False", "True"],
        },
        columns=["Feature 1", "Feature 2", "Miss & Class"],
    )
    bin_number = 2

    result = misclass_rate_feature(
        dataset, dataset_misclass, feature="Feature 1", bins=bin_number
    )
    assert len(result) == 3
    assert len(result[0]) == len(result[1]) + 1
    assert len(result[1]) == bin_number

    assert max(result[1]) <= 1.0
    assert min(result[1]) >= 0.0


def test_show_misclass_rate_feature():
    pass


def test_show_misclass_rates_features():
    pass


def test_compute_tiles():
    dataset = pd.DataFrame(
        {
            "Feature 1": [1, 2.5, 3.5, 5],
            "Feature 2": [1, 6, 8, 9],
            "Miss & Class": ["True", "False", "> Prediction hit", "True"],
        },
        columns=["Feature 1", "Feature 2", "Miss & Class"],
    )

    selected_feature = "Feature 1"
    tile_number = 2

    edges_bins = compute_tiles(dataset, feature=selected_feature, tiles=tile_number)

    assert edges_bins[0] == min(dataset[selected_feature])
    assert edges_bins[-1] == max(dataset[selected_feature])
    assert len(edges_bins) == tile_number + 1


def test_show_tiles_feature():
    pass


def test_show_tiles_features():
    pass
