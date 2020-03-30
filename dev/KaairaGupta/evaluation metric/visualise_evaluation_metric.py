import matplotlib.pyplot as plt
import numpy as np


def visualise_evaluation_metric(table):

    """
        This function plot a metric against the values of some parameter with repeated runs to assess variability.
        Input: 
            table: array with columns (x, y1, y2, ..., yk), i.e. multiple y values for each x.
        Output: 
            Plots showing:
                Average y value vs x with the spread of y-values represented by min-max band of y.
                Violin plots showing variability of y.
    """

    table = np.array(table)

    x = (table[:, 0]).astype(int)

    ### Plot Mean and Min-Max Band: Begin ###

    _mean = np.empty(len(x))
    _min = np.empty(len(x))
    _max = np.empty(len(x))

    for i in range(len(table)):
        r = table[i]

        _mean[i] = np.mean(r[1:])
        _min[i] = np.min(r[1:])
        _max[i] = np.max(r[1:])

    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(x, _mean, label="Mean", linewidth=3, color="g")
    ax.plot(x, _min, label="Min", linewidth=0.5, color="y")
    ax.plot(x, _max, label="Max", linewidth=0.5, color="r")
    ax.fill_between(x, _min, _max, alpha=0.1, label="Min-Max Band")

    # Axes Labels
    plt.xlabel("X")
    plt.ylabel("Y")

    # Figure Title
    fig.suptitle("Mean and Min-Max Band", fontweight="bold")

    plt.legend()
    plt.show()

    ### Plot Mean and Min-Max Band: End ###

    ### Violin Plots: Begin ###

    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Create the violin plot
    ax.violinplot(table[:, 1:].tolist(), x.tolist(), showmeans=True)

    # Axes Labels
    plt.xlabel("X")
    plt.ylabel("Y")

    # Figure Title
    fig.suptitle("Violin Plots", fontweight="bold")

    plt.show()

    ### Violin Plots: End ###
