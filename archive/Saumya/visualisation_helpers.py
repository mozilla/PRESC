import pandas as pd
import numpy as np

from itertools import cycle
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import time

matplotlib.style.use("ggplot")
sns.set_style("white")
pylab.rcParams["figure.figsize"] = 12, 8


def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={"shrink": 0.9},
        ax=ax,
        annot=True,
        linewidths=0.1,
        vmax=1.0,
        linecolor="white",
        annot_kws={"fontsize": 12},
    )

    plt.title("Pearson Correlation of Features", y=1.05, size=15)


def accuracy_barplot(MLA_compare):

    sns.barplot(x="MLA Test Accuracy Mean", y="MLA Name", data=MLA_compare, color="m")

    plt.title("Machine Learning Algorithms' Accuracy Score \n")
    plt.xlabel("Accuracy Score (%)")
    plt.ylabel("Algorithm")


# is not adapted to work with binary data, however, there is a workaround by tweaking the dimensions of test_y and y_score
def multiclass_roc(n_classes, test_y, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def plot_roc(fpr, tpr, roc_auc, n_classes=4, lw=3):
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    # Four colours for the four classes
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=3,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One vs rest ROC curves")
    plt.legend(loc="lower right")
    plt.show()


def print_roc_auc_score(classifier, test_X, test_y):
    y_prob = classifier.predict_proba(test_X)

    macro_roc_auc_ovo = roc_auc_score(
        test_y, y_prob, multi_class="ovo", average="macro"
    )
    weighted_roc_auc_ovo = roc_auc_score(
        test_y, y_prob, multi_class="ovo", average="weighted"
    )
    macro_roc_auc_ovr = roc_auc_score(
        test_y, y_prob, multi_class="ovr", average="macro"
    )
    weighted_roc_auc_ovr = roc_auc_score(
        test_y, y_prob, multi_class="ovr", average="weighted"
    )
    print(
        "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
    )
    print(
        "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
    )


def multiclass_precision_recall(n_classes, test_y, y_score):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(test_y[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(test_y[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        test_y.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        test_y, y_score, average="micro"
    )
    print(
        "Average precision score, micro-averaged over all classes: {0:0.2f}".format(
            average_precision["micro"]
        )
    )
    return precision, average_precision, recall


def plot_precision_recall(n_classes, precision, average_precision, recall):
    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall["micro"], precision["micro"], color="gold", lw=2)
    lines.append(l)
    labels.append(
        "micro-average Precision-recall (area = {0:0.2f})"
        "".format(average_precision["micro"])
    )

    for i, color in zip(range(n_classes), colors):
        (l,) = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(
            "Precision-recall for class {0} (area = {1:0.2f})"
            "".format(i, average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall curves")
    plt.legend(lines, labels, loc=(0, -0.38), prop=dict(size=14))

    plt.show()
