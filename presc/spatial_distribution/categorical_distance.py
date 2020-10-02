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


""" Compute the number d of categorical attributes the objects have """


""" Compute  the number of times the attribute Ak takes a particular value x """

""" Compute the sample probability that the categrical attribute Ak takes a value x
    (the number of times the attribute Ak takes a value x over the number of objects) """

""" Given a point, a dataset and a defined metric compute the averge distance of that point to every
other point in the data set"""


def distance_to_data(dpoint, data, metric, sample_size=1000):
    # assuming data is a pandas dataframe
    sampled_indexes = random.sample(range(len(data)), sample_size)
    distance = 0
    for i in sampled_indexes:
        distance = distance + metric(dpoint, data.iloc[i])

    return distance / sample_size


""" Implementing Overlap """


def plot_distribution(data, metric, sample_size=1000):
    distance_array = np.empty(sample_size)
    sampled_indexes = random.sample(
        range(len(data)), sample_size
    )  # generate the random indexes
    j = 0
    for i in sampled_indexes:
        distance_array[j] = distance_to_data(
            data.iloc[i], data.drop(data.index[i]), metric
        )  # compute the distance to every other point
        j = j + 1

    sns.set()
    plt.hist(distance_array, bins=20)
    plt.show()
    pass


def create_distance(data, metric, array, index):
    array[index] = distance_to_data(
        data.iloc[index], data.drop(data.index[index]), metric
    )
    pass


def overlap(dpoint1, dpoint2):
    """Computes the overlap similarity between two data point that have only attributes
    of categorical nature"""
    # Let's first assume they are the same lenght

    attributesd1 = list(dpoint1.index)
    attributesd2 = list(dpoint2.index)
    # list_dpoint1=np.array(dpoint1)
    # list_dpoint2=np.array(dpoint2)
    if attributesd1 == attributesd2:
        overlap_score = len(set(dpoint1) & set(dpoint2)) / len(attributesd1)

        return overlap_score
    """
        for i in range(0,len(attributesd1)):
            if list_dpoint1[i] == list_dpoint2[i]:
                overlap_score += 1/len(attributesd1)
                print(list_dpoint1[i])
            pass
        pass
    """

    pass


def goodall2(dpoint1, dpoint2):
    attributesd1 = list(dpoint1.index)
    # attributesd2 = list(dpoint2.index)
    goodall2_score = 0
    for attribute in attributesd1:
        attribute_score = 0
        if dpoint1[attribute] == dpoint2[attribute]:  # if the values are the same
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
    print(value_counts)
    return value_counts


def modified_frequency(counts):
    data_len = len(data)
    estimated_freq = (counts * (counts - 1)) / ((data_len) * (data_len - 1))
    return estimated_freq


def times_value_happens(data, attribute, value):
    # filt = data.groupby(attribute)
    filt = data[attribute].value_counts()
    print(filt[value])
    pass


def main():
    a = data.iloc[-1]
    b = data.iloc[-30]
    # overlap(a, b)
    print(goodall3(a, b))

    # print(counts)
    print("go")
    # print(distance_to_data(a,data,goodall3))
    # print(distance_to_data(a,data,overlap))
    #    c=data.loc[["edibility","cap-shape"]]
    print(data_transpose)
    print(data_transpose.loc[0])
    # plot_distribution(data,goodall3)
    #    empirical_frequency(a.index[4], a[a.index[4]])

    pass


main()
