"""This program seeks to compare the prediction results of two or more machine learning models, 
when they are trained and used on the same dataset. It is the first step in guiding the choice of 
an optimal model for the dataset of interest. """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def class_probs(labels, estimators, data_test):

    preds = []
    prob0s = []
    prob1s = []
    for clf in estimators:
        pred = clf.predict_proba(data_test)
        preds.append(pred)

        prob0 = np.mean(pred[:, 0])
        prob0s.append(prob0)

        prob1 = np.mean(pred[:, 1])
        prob1s.append(prob1)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(6, 7))
    rects1 = ax.bar(x - width / 2, prob0s, width, label="Class0")
    rects2 = ax.bar(x + width / 2, prob1s, width, label="Class1")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Mean probability")
    ax.set_title("Probability by estimator and class")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            height = round(height, 2)
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()
