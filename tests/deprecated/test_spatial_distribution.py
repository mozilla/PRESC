import random as rd
import pandas as pd
import numpy as np
import string
import presc.deprecated.spatial_distribution.categorical_distance as catd
import pytest


@pytest.fixture
def random_catd_analysis():
    toy_data = []
    column_names = []
    lower_upper_alphabet = string.ascii_letters
    for i in range(0, 9):
        name = "feature_" + str(i)
        column_names.append(name)
    column_names.append("label")

    for data_row in range(0, 100):
        data_entry = []
        for feature in range(0, 9):
            data_entry.append(rd.choice(lower_upper_alphabet))

        data_entry.append(rd.choice([True, False]))  # appending the classification row
        data_entry = tuple(data_entry)
        toy_data.append(data_entry)

    df = pd.DataFrame(toy_data, columns=column_names)

    y_pred = np.random.randint(2, size=len(df))
    print(y_pred)

    catd_random = catd.SpatialDistribution(df, df["label"], y_pred)
    return catd_random


@pytest.fixture
def catd_analysis():
    column_names = []
    for i in range(8):
        name = "feature_" + str(i)
        column_names.append(name)
    column_names.append("label")

    toy_data = [
        ("p", "t", "x", "v", "q", "p", 1, 2, 3),
        ("e", "y", "x", "y", "q", "p", 3, 4, 5),
        ("p", "x", "s", "y", "t", "p", 7, 8, 9),
        ("e", "x", "s", "y", "t", "a", 9, 10, 11),
        ("e", "b", "s", "w", "t", "a", 11, 2, 3),
        ("p", "x", "y", "w", "t", "p", 8, 9, 10),
    ]

    df = pd.DataFrame(toy_data, columns=column_names)
    y_pred = ["0", "1", "0", "1", "0", "1"]
    y_true = ["0", "1", "1", "1", "0", "0"]
    catd_analysis = catd.SpatialDistribution(df, y_pred, y_true)

    return catd_analysis


def test_goodall2(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(-1)
    dpoint2 = catd_analysis.get_datapoint(-2)
    hand_calculated_value = 0.3222222
    output_value = catd_analysis.goodall2(dpoint1, dpoint2)
    np.testing.assert_approx_equal(hand_calculated_value, output_value)


def test_goodall3(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(0)
    dpoint2 = catd_analysis.get_datapoint(1)
    hand_calculated_value = (1 / 6) * (((2 * (14 / 15))) + (1 - (6.0 / 15)))
    output_value = catd_analysis.goodall3(dpoint1, dpoint2)
    print(hand_calculated_value)
    print(output_value)
    np.testing.assert_approx_equal(hand_calculated_value, output_value)


def test_lin(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(2)
    dpoint2 = catd_analysis.get_datapoint(3)
    hand_calculated_value = (
        2 * np.log(1)
        + 2 * np.log((3 / 6) * (3 / 6) * (3 / 6) * (4 / 6))
        + 2 * np.log((4 / 6) + (2 / 6))
    )
    lin_weight = (
        (np.log(3 / 6) + np.log(3 / 6))
        + 2 * np.log(3 / 6)
        + 2 * np.log(3 / 6)
        + 2 * np.log(3 / 6)
        + 2 * (np.log(4 / 6))
        + (np.log(4 / 6) + np.log(2 / 6))
    )
    hand_calculated_value = (1 / lin_weight) * hand_calculated_value
    output_value = catd_analysis.lin(dpoint1, dpoint2)
    np.testing.assert_approx_equal(hand_calculated_value, output_value)


def test_overlap(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(4)
    dpoint2 = catd_analysis.get_datapoint(5)
    hand_calculated_value = 2 / 6
    output_value = catd_analysis.overlap(dpoint1, dpoint2)
    np.testing.assert_approx_equal(hand_calculated_value, output_value)


""" Distances should be symmetric """


def test_goodall2_symmetry(random_catd_analysis):
    dpoint1 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    dpoint2 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )

    goodall2_left = random_catd_analysis.goodall2(dpoint1, dpoint2)
    goodall2_right = random_catd_analysis.goodall2(dpoint1, dpoint2)
    np.testing.assert_approx_equal(goodall2_left, goodall2_right)


def test_goodall3_symmetry(random_catd_analysis):
    dpoint1 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    dpoint2 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    goodall3_left = random_catd_analysis.goodall3(dpoint1, dpoint2)
    goodall3_right = random_catd_analysis.goodall3(dpoint1, dpoint2)
    np.testing.assert_approx_equal(goodall3_left, goodall3_right)


def test_lin_symmetry(random_catd_analysis):
    dpoint1 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    dpoint2 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    lin_left = random_catd_analysis.lin(dpoint1, dpoint2)
    lin_right = random_catd_analysis.lin(dpoint1, dpoint2)
    np.testing.assert_approx_equal(lin_left, lin_right)


def test_overlap_symmetry(random_catd_analysis):
    dpoint1 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    dpoint2 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    overlap_left = random_catd_analysis.overlap(dpoint1, dpoint2)
    overlap_right = random_catd_analysis.overlap(dpoint1, dpoint2)
    np.testing.assert_approx_equal(overlap_left, overlap_right)


def test_array_of_distance_intersection(random_catd_analysis):
    """The intersection of the indexes of the points returned in the tuple list
    must have a set intersection of len = 0"""
    dpoint1 = random_catd_analysis.get_datapoint(
        rd.randint(0, random_catd_analysis.get_data_len() - 1)
    )
    misclass_only = random_catd_analysis.array_of_distance(
        dpoint1, "lin", misclass_only=True
    )
    trueclass_only = random_catd_analysis.array_of_distance(
        dpoint1, "lin", correct_only=True
    )
    miss_index = [tuple[1] for tuple in misclass_only]
    diss_index = [tuple[1] for tuple in trueclass_only]
    set_intersection = set(miss_index) & set(diss_index)

    assert len(set_intersection) == 0


def test_plot_distance_point_histogram(catd_analysis):
    dpoint = catd_analysis.get_datapoint(1)
    other_metrics = ["overlap", "goodall2"]
    catd_analysis.plot_distance_to_point_histogram(
        dpoint, "lin", other_metrics=other_metrics, bar_width=5
    )


def test_avg_distance_histogram(catd_analysis):
    catd_analysis.plot_avg_distance_histogram("lin", histo_sample=1, distance_sample=1)


def test_plot_distance_scatterplot(catd_analysis):
    catd_analysis.plot_distance_scatterplot(
        "lin", "l2_norm", scatter_sample=1, distance_sample=1
    )


def test_plot_distance_misclassifed(catd_analysis):
    catd_analysis.plot_distance_misclassified("lin", "l2_norm", distance_sample=1)
