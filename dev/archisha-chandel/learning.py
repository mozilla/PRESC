__author__ = 'Archisha Chandel'
__copyright__ = 'Copyright 2020, mozilla/PRESC'

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    make_scorer,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)


clf = RandomForestClassifier(n_jobs=-1)

param_grid = {
    "min_samples_split": [3, 5, 7, 13],
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 10],
    "max_features": [3, 5, 10, 13],
}

# The `scorers` dictionary can be used as the `scoring` argument in `GridSearchCV`.
# When multiple scores are passed, `GridSearchCV.cv_results_` will return scoring metrics for each of the score types provided.
scorers = {
    "precision_score": make_scorer(precision_score),
    "recall_score": make_scorer(recall_score),
    "accuracy_score": make_scorer(accuracy_score),
}


def grid_search_wrapper(refit_score, X_train, y_train, X_test, y_test):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """

    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(
        clf,
        param_grid,
        scoring=scorers,
        refit=refit_score,
        cv=skf,
        return_train_score=True,
        n_jobs=4,
    )
    grid_search.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid_search.predict(X_test.values)

    print("Best params for {}".format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print(
        "\nConfusion matrix of Random Forest optimized for {} on the test data:".format(
            refit_score
        )
    )
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            columns=["pred_neg", "pred_pos"],
            index=["neg", "pos"],
        )
    )
    return grid_search


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """

    return [1 if y >= t else 0 for y in y_scores]


def precision_recall_threshold(t, y_scores, y_test, thresholds, p, r):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """

    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred_adj),
            columns=["pred_neg", "pred_pos"],
            index=["neg", "pos"],
        )
    )

    # plot the curve
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color="b", alpha=0.2, where="post")
    plt.fill_between(r, p, step="post", alpha=0.2, color="b")
    plt.ylim([0.5, 1.01])
    plt.xlim([0.5, 1.01])
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], "^", c="k", markersize=15)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="best")


def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8, 8))
    plt.title("ROC Curve")
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="best")
