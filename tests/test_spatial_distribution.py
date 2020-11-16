import random as rd
import pandas as pd
import numpy as np
import string
import presc.spatial_distribution.categorical_distance as catd
import pytest


@pytest.fixture
def catd_analysis():
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

        data_entry.append(rd.choice([0, 1]))
        data_entry = tuple(data_entry)
        toy_data.append(data_entry)

    df = pd.DataFrame(toy_data, columns=column_names)

    y_pred = np.random.randint(2, size=len(df))
    print(y_pred)

    catd_analysis = catd.SpatialDistribution(df, df["label"], y_pred)
    return catd_analysis


# Distances should be symmetric
def test_lin(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )
    dpoint2 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )

    assert catd_analysis.lin(dpoint1, dpoint2) == catd_analysis.lin(dpoint2, dpoint1)


def test_overlap(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )
    dpoint2 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )

    assert catd_analysis.overlap(dpoint1, dpoint2) == catd_analysis.overlap(
        dpoint2, dpoint1
    )


def test_goodall2(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )
    dpoint2 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )

    assert catd_analysis.goodall2(dpoint1, dpoint2) == catd_analysis.goodall2(
        dpoint2, dpoint1
    )


def test_goodall3(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )
    dpoint2 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )
    assert catd_analysis.goodall3(dpoint1, dpoint2) == catd_analysis.goodall3(
        dpoint2, dpoint1
    )


def test_distance_histogram(catd_analysis):
    catd_analysis.plot_distance_histogram("overlap", histo_sample=1, distance_sample=1)


def test_plot_distance_scatterplot(catd_analysis):
    catd_analysis.plot_distance_scatterplot(
        "lin", "goodall2", scatter_sample=1, distance_sample=1
    )


def test_plot_distance_misclasifed(catd_analysis):
    catd_analysis.plot_distance_misclasified("lin", "goodall2", distance_sample=1)


def test_array_of_distances(catd_analysis):
    dpoint1 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )
    array1 = catd_analysis.array_of_distance(dpoint1, "lin", 10)
    array2 = catd_analysis.array_of_distance(dpoint1, "lin", 10)
    assert array1 == array2


def test_knearest_points(catd_analysis):
    k = 5
    dpoint1 = catd_analysis.get_datapoint(
        rd.randint(0, catd_analysis.get_data_len() - 1)
    )
    catd_analysis.plot_knearest_points(dpoint1, "goodall3", "overlap", "goodall2", k)
