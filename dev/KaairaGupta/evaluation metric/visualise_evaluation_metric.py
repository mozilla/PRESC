import matplotlib.pyplot as plt
import numpy as np


def visualise_evaluation_metric(table):

    """
        This function plot a metric against the values of some parameter with repeated runs to assess variability.
        Input: Table (numpy array) with columns (x, y1, y2, ..., yk), i.e. multiple y values for each x.
        Output: Plots showing the average y value vs x with the spread of the y-values represented by min-max and standard-deviation of y.
    """

    x = table[0]

    _mean = np.empty(len(x))
    _min = np.empty(len(x))
    _max = np.empty(len(x))
    _std = np.empty(len(x))

    table = table.transpose()

    for i in range(len(table)):
        r = table[i]

        _mean[i] = np.mean(r[1:])
        _min[i] = np.min(r[1:])
        _max[i] = np.max(r[1:])
        _std[i] = np.std(r[1:])

    plt.plot(x, _mean, label="Mean", linewidth=2, color="g")
    plt.plot(x, _min, label="Min", linewidth=1, color="y")
    plt.plot(x, _max, label="Max", linewidth=1, color="r")
    plt.fill_between(
        x, _mean - (_std / 2), _mean + (_std / 2), alpha=0.3, label="Standard Deviation"
    )

    plt.legend()
    plt.show()
