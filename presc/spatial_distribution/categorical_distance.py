import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns

data = pd.read_csv("C:/Users/castromi/Documents/GitHub/PRESC/datasets/mushrooms.csv")
columns_name = list(data.columns)
counts_dictionary = {}
data_transpose = data.transpose()

for attribute in columns_name:
    counts_dictionary[attribute] = data[attribute].value_counts()


""" Given a point, a dataset and a defined metric compute the averge distance of that point to every
other point in the data set"""


def distance_to_data(dpoint, data, metric, sample_size=100):
    # assuming data is a pandas dataframe
    sampled_indexes = random.sample(range(len(data)), sample_size)
    distance = 0
    for i in sampled_indexes:
        distance = distance + metric(dpoint, data.iloc[i])

    return distance / sample_size


""" Implementing Overlap """


def plot_histogram(data, metric, histo_sample=1000, dist_sample=100):
    distance_array = np.empty(histo_sample)
    sampled_indexes = random.sample(
        range(len(data)), histo_sample
    )  # generate the random indexes
    j = 0
    for i in sampled_indexes:
        distance_array[j] = distance_to_data(
            data.iloc[i], data.drop(data.index[i]), metric, dist_sample
        )
        j = j + 1

    sns.set()
    plt.hist(distance_array, bins=20)
    plt.show()
    pass


def plot_distance_scatterplot(data, metric1, metric2, sample_size=100):
    distance_array_metric1 = np.empty(sample_size)
    distance_array_metric2 = np.empty(sample_size)
    sampled_indexes = random.sample(
        range(len(data)), sample_size
    )  # generate the random indexes
    j = 0
    for i in sampled_indexes:
        distance_array_metric1[j] = distance_to_data(
            data.iloc[i], data.drop(data.index[i]), metric1
        )
        distance_array_metric2[j] = distance_to_data(
            data.iloc[i], data.drop(data.index[i]), metric2
        )
        print(j)
        j = j + 1

    sns.set()
    fig, ax = plt.subplots()

    ax.scatter(distance_array_metric1, distance_array_metric2)
    ax.set_xlabel(metric1.__name__ + " distance")
    ax.set_ylabel(metric2.__name__ + " distance")
    plt.show()
    pass


def overlap(dpoint1, dpoint2):
    """Computes the overlap similarity between two data point that have only attributes
    of categorical nature"""

    attributesd1 = list(dpoint1.index)
    attributesd2 = list(dpoint2.index)

    if attributesd1 == attributesd2:  # check if they have the same attributes
        overlap_score = len(set(dpoint1) & set(dpoint2)) / len(attributesd1)
        return overlap_score
    pass


def goodall2(dpoint1, dpoint2):
    attributesd1 = list(dpoint1.index)
    # attributesd2 = list(dpoint2.index)
    goodall2_score = 0
    for attribute in attributesd1:
        attribute_score = 0
        if (
            dpoint1[attribute] == dpoint2[attribute]
        ):  # check if the values  are the same
            counts = data[attribute].value_counts()
            for count in counts:
                attribute_score = modified_frequency(count) + attribute_score

            goodall2_score = (1 - attribute_score) + goodall2_score

    return goodall2_score / len(attributesd1)

    pass


def goodall3(dpoint1, dpoint2):
    attributesd1 = list(dpoint1.index)
    # attributesd2 = list(dpoint2.index)
    goodall3_score = 0
    for attribute in attributesd1:
        attribute_score = 0
        value_point1 = dpoint1[attribute]
        value_point2 = dpoint2[attribute]
        if value_point1 == value_point2:
            count = counts_dictionary[attribute][value_point1]
            attribute_score = modified_frequency(count)
            goodall3_score = (1 - attribute_score) + goodall3_score

    return goodall3_score / len(attributesd1)
    pass


def value_count(attribute, value):
    filt = data[attribute].value_counts()
    value_counts = filt[value]
    return value_counts


def modified_frequency(counts):
    data_len = len(data)
    estimated_freq = (counts * (counts - 1)) / ((data_len) * (data_len - 1))
    return estimated_freq


def main():
    # a = data.iloc[-1]
    # b = data.iloc[-10]

    # print(counts)
    print("go")
    # print(distance_to_data(a,data,goodall3))
    # print(distance_to_data(a,data,overlap))
    #    c=data.loc[["edibility","cap-shape"]]
    # print(list(data_transpose.columns))
    # atranspose = data_transpose[0].to_frame()
    # btranspose = data_transpose[1].to_frame()

    # print(a[a.isin(b)])
    # print(b)
    plot_histogram(data, goodall3, histo_sample=1000)
    # plot_distance_scatterplot(data,overlap,goodall3,sample_size=1000)
    #    empirical_frequency(a.index[4], a[a.index[4]])

    pass


main()
