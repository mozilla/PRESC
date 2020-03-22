# Learning From Misclassifications

from dataloader import get_x_y
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import validation_curve, learning_curve, ShuffleSplit

def plot_learning_curve(
    estimator,
    title,
    x,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Plotting Learning Curves

    Parameters:
        estimator : trained classification moel
        title : str, title of the graph
        x : array-like, shape(n_samples, n_features)
        y : of length n_features
        axes : axes
        ylim : array-like
        cv : int
        n_jobs : int
        train_sizes : array-like

    Returns:
        void
    """

    if axes is None:
        _, axes = plt.subplots()

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        x,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def plot_lc(clf):
    """
    Plots Learning Curves for the giveen classification model.

    Parameters:
        clf : trained classification model

    Returns:
        void
    """

    x, y = get_x_y()
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    title = "Learning Curves"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_learning_curve(clf, title, x, y, axes=axes, cv=cv, n_jobs=4)
    plt.show()

def mis_visual(clf, x_train, x_val, x_test, y_train, y_val, y_test):
    """
    Plots Bar Graph for the counts of misclassified examples for train, test and validation samples.

    Parameters:
        clf : trained classification model
        x_train : array-like , shape(n_train_samples, n_features)
        x_val : array-like , shape(n_val_samples, n__features)
        x_test : array-like , shape(n_test_samples, n_features)
        y_train : of length n_train_samples
        y_val : of length n_val_samples
        y_test : of length n_test_samples

    Returns:
        void
    """

    y_val_score = clf.predict(x_val)
    y_test_score = clf.predict(x_test)
    y_train_score = clf.predict(x_train)

    # calcualting errors
    val_err = np.array(np.where(y_val != y_val_score))
    test_err = np.array(np.where(y_test != y_test_score))
    train_err = np.array(np.where(y_train != y_train_score))

    # misclassified y values
    y_val_err = y_val_score[val_err]
    y_test_err = y_test_score[test_err]
    y_train_err = y_train_score[train_err]
    mis_val_counts = np.unique(y_val_err, return_counts=True)
    mis_test_counts = np.unique(y_test_err, return_counts=True)
    mis_train_counts = np.unique(y_train_err, return_counts=True)

    #  plotting bar graph errors
    n_groups = 3
    false_class1 = (mis_train_counts[1][0], mis_val_counts[1][0], mis_test_counts[1][0])
    false_class2 = (mis_train_counts[1][1], mis_val_counts[1][1], mis_test_counts[1][1])

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(
        index, false_class1, bar_width, alpha=opacity, color="g", label="Class1"
    )

    rects2 = plt.bar(
        index + bar_width,
        false_class2,
        bar_width,
        alpha=opacity,
        color="purple",
        label="Class2",
    )

    plt.ylabel("Misclassified Count")
    plt.title("Misclassified Values")
    plt.xticks(index + bar_width, ("y train", "y validation", "y test"))
    plt.legend()

    plt.tight_layout()
    plt.show()

def err_score(clf, x_train, x_val, x_test, y_train, y_val, y_test):
    """
    The function calculates error % for train, validation and test samples.

    Parameters:
        clf : trained classification model
        x_train : array-like , shape(n_train_samples, n_features)
        x_val : array-like , shape(n_val_samples, n__features)
        x_test : array-like , shape(n_test_samples, n_features)
        y_train : of length n_train_samples
        y_val : of length n_val_samples
        y_test : of length n_test_samples

    Returns:
        errors : dictionary
        
    """

    # calculating errors
    train_err = (1 - clf.score(x_train, y_train)) * 100
    val_err = (1 - clf.score(x_val, y_val)) * 100
    test_err = (1 - clf.score(x_test, y_test)) * 100

    errors = {
        "train-error %": train_err,
        "validation-error %": val_err,
        "test-error %": test_err,
    }

    return errors
