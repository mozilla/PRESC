import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns

data = pd.read_csv("C:/Users/castromi/Documents/GitHub/PRESC/datasets/mushrooms.csv")

""" Compute the number d of categorical attributes the objects have """


""" Compute  the number of times the attribute Ak takes a particular value x """

""" Compute the sample probability that the categrical attribute Ak takes a value x
    (the number of times the attribute Ak takes a value x over the number of objects) """

""" Given a point, a dataset and a defined metric compute the averge distance of that point to every
other point in the data set"""


def distance_to_data(dpoint, data, metric):
    # assuming data is a pandas dataframe
    distance = 0
    for i in range(0, len(data)):
        distance = distance + metric(dpoint, data.iloc[i])

    return distance / len(data)


""" Implementing Overlap """


def plot_distribution(data, metric, sample=100):
    distance_array = np.empty(sample)
    sampled_indexes = random.sample(
        range(len(data)), sample
    )  # sample 20 non-repeating indexes
    j = 0
    print(sampled_indexes)
    for i in sampled_indexes:
        distance_array[j] = distance_to_data(
            data.iloc[i], data.drop(data.index[i]), metric
        )  # compute the distance to every other point
        print(j)
        j = j + 1
    print(distance_array)
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


"""
def goodall2(dpoint1,dpoint2):
    attributesd1=list(dpoint1.index)   #get the labels
    attributesd2=list(dpoint2.index)


    fk = data[attributed1[0]
    #list_dpoint1=np.array(dpoint1)
    #list_dpoint2=np.array(dpoint2)
    if attributesd1 == attributesd2:
        overlap = (set(dpoint1) & set(dpoint2))/len(attributesd1)




    pass
"""


def times_value_happens(data, attribute, value):
    # filt = data.groupby(attribute)
    filt = data[attribute].value_counts()
    print(filt[value])
    pass


def main():
    a = data.iloc[0]
    print(a.index[1])
    b = data.iloc[-30]
    print(list(a.index))
    print(len(data))
    print(type(a))
    print(type(b))
    print(len(a))
    print(a[a.index[0]])
    print(b[b.index[22]])

    overlap(a, b)
    #    c=data.loc[["edibility","cap-shape"]]

    # plot_distribution(data,overlap,400)
    times_value_happens(data, a.index[1], a[a.index[1]])

    pass


main()
