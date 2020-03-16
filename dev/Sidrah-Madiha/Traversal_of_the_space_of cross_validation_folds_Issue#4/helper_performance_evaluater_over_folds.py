import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd


def performance_evaluater_over_folds(classifier, no_of_cv, X, y):
    """returns table (type dataframe) containing folds and corresponding average cross validation score
    inputs:
    classifier : classifier/model/estimator
    no_of_cv : range of folds you want to generate table for
    X: training data
    y: target label"""

    scores = list()
    range_of_cv = list(range(2, no_of_cv + 1))
    for i in range_of_cv:
        scores_avg = np.mean(cross_val_score(classifier, X, y, cv=i))
        scores.append(scores_avg)
    table = pd.DataFrame({"No. of folds": range_of_cv, "Average metric Score": scores})
    table.set_index("No. of folds", inplace=True)
    return table


def visualising_performance_evaluater_over_folds(table):
    """ displays a bar plot that shows cross validation score for each fold
    inputs:
    table: dataframe that has 2 columns: 'No. of folds' and 'Average metric Score' """

    folds = table.index.values.tolist()
    score = table["Average metric Score"].values.tolist()
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(folds, score)
    plt.xticks(folds)
    ax.set_title("Scores by No. of Folds")
    plt.ylabel("Average Metric Score")
    plt.xlabel("No. of Folds")
    ax.axhline(
        np.mean(score),
        label="Mean score = {:0.3f}".format(np.mean(score)),
        linestyle="--",
        linewidth=0.3,
    )
    plt.legend(loc="upper right")
    axes = plt.gca()
    ymin = min(score)
    ymax = max(score)
    axes.set_ylim([ymin - (ymin * 0.001), ymax + (ymax * 0.001)])
    plt.show()
