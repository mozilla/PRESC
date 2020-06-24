import numpy as np

np.random.seed(0)

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib import cm


def compare_test_sample_classification_for_models(
    model_list, model_name, X_train, y_train, X_test, y_test
):
    """ visualizes two graphs, first graph shows spread of class probablities for each model, other shows no. of missclassifed datapoints in 
    of each class in each model"""

    plt.figure(figsize=(9, 9))

    missclassified_0 = []
    missclassified_1 = []

    ax2 = plt.subplot2grid((3, 1), (2, 0))
    for clf, name in zip(model_list, model_name):
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
            zero_miss, one_miss = missclassify(prob_pos, y_test)
            missclassified_0.append(zero_miss)
            missclassified_1.append(one_miss)

        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            zero_miss, one_miss = missclassify(prob_pos, y_test)
            missclassified_0.append(zero_miss)
            missclassified_1.append(one_miss)

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    ax2.set_title("Spread of Class Probablities of Each Model")

    plt.tight_layout()
    plot_missclassify(model_name, missclassified_0, missclassified_1)


def missclassify(prob_pos, y_test):
    """ Returns no of missclassified points in class: 0 and class: 1"""
    zero_miss = 0
    one_miss = 0
    for each, label in zip(prob_pos, y_test):
        if each > 0.5 and label == 0:
            zero_miss = zero_miss + 1
        elif each < 0.5 and label == 1:
            one_miss = one_miss + 1
    return zero_miss, one_miss


def plot_missclassify(model_name, missclassified_0, missclassified_1):
    """ to visualize missclassified points in each class for each model"""
    x = np.arange(len(model_name))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    rects1 = ax.bar(
        x - width / 2, missclassified_0, width, label="Class: 0 missclassify"
    )
    rects2 = ax.bar(
        x + width / 2, missclassified_1, width, label="Class: 1 missclassify"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("No. of misscalssified points")
    ax.set_title("Missclassified point in each class for each model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_name)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    fig.tight_layout()

    plt.show()


def autolabel(ax, rects):
    """ for labelling bars on graph"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
