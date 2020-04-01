# This file maps the evaluation metrics for Repition of CV Splits
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn import preprocessing
from classifiers import Classifier
from evaluation import evaluate
from dataloader import get_x_y
import random
import matplotlib.pyplot as plt
from IPython.display import HTML
from sklearn.ensemble import RandomForestClassifier

def perform_repeated_cv(model, n_reps=50):
    """
	Performs repeated cross validation on the dataset.

	Parameters:
		model : trained classification model
		n_reps : int, optional(default = 50)

	Returns;
		accuracy_scores : array-like
		precision_scores : array-like
		recall_scores : array-like
	"""

    # set random seed for repeartability
    random.seed(1)

    # obtaining x and y
    x, y = get_x_y()
    x = np.array(x)
    y = np.array(y)

    # perform repeated cross validation
    accuracy_scores = np.zeros(n_reps)
    precision_scores = np.zeros(n_reps)
    recall_scores = np.zeros(n_reps)

    for u in range(n_reps):

        # randomly shuffle the dataset
        indices = np.arange(x.shape[0])
        # print(indices.shape)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]  # dataset has been randomly shuffled

        # initialize vector to keep predictions from all folds of the cross-validation
        y_predicted = np.zeros(y.shape)

        # performing cross validation
        kf = KFold(n_splits=5, random_state=142)
        for train, test in kf.split(x):

            # split the dataset into training and testing
            x_train = x[train]
            x_test = x[test]
            y_train = y[train]
            y_test = y[test]

            # standardization
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            # train model
            clf = model
            clf.fit(x_train, y_train)

            # make predictions on the testing set
            y_predicted[test] = clf.predict(x_test)

        # record scores
        accuracy_scores[u], precision_scores[u], recall_scores[u], _, _ = evaluate(
            clf, x_test, y_test
        )

    # return all scores
    return accuracy_scores, precision_scores, recall_scores

def plot_cv_repeat(clf):
    """
	The function plots number repeatations vs the evalution metrics - accuracy, precision, recall.

	Pararmeters:
		clf : trained classification model

	Returns:
		non 
	"""

    accuracy_scores, precision_scores, recall_scores = perform_repeated_cv(clf)

    # plot results from the 50 repetitions
    fig, axes = plt.subplots(3, 1)

    axes[0].plot(100 * accuracy_scores, color="xkcd:cherry", marker="o")
    axes[0].set_xlabel("Repetition")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_facecolor((1, 1, 1))
    axes[0].spines["left"].set_color("black")
    axes[0].spines["right"].set_color("black")
    axes[0].spines["top"].set_color("black")
    axes[0].spines["bottom"].set_color("black")
    axes[0].spines["left"].set_linewidth(0.5)
    axes[0].spines["right"].set_linewidth(0.5)
    axes[0].spines["top"].set_linewidth(0.5)
    axes[0].spines["bottom"].set_linewidth(0.5)
    axes[0].grid(linestyle="--", linewidth="0.5", color="grey", alpha=0.5)

    axes[1].plot(100 * precision_scores, color="xkcd:royal blue", marker="o")
    axes[1].set_xlabel("Repetition")
    axes[1].set_ylabel("Precision(%)")
    axes[1].set_facecolor((1, 1, 1))
    axes[1].spines["left"].set_color("black")
    axes[1].spines["right"].set_color("black")
    axes[1].spines["top"].set_color("black")
    axes[1].spines["bottom"].set_color("black")
    axes[1].spines["left"].set_linewidth(0.5)
    axes[1].spines["right"].set_linewidth(0.5)
    axes[1].spines["top"].set_linewidth(0.5)
    axes[1].spines["bottom"].set_linewidth(0.5)
    axes[1].grid(linestyle="--", linewidth="0.5", color="grey", alpha=0.5)

    axes[2].plot(100 * recall_scores, color="xkcd:emerald", marker="o")
    axes[2].set_xlabel("Repetition")
    axes[2].set_ylabel("Recall (%)")
    axes[2].set_facecolor((1, 1, 1))
    axes[2].spines["left"].set_color("black")
    axes[2].spines["right"].set_color("black")
    axes[2].spines["top"].set_color("black")
    axes[2].spines["bottom"].set_color("black")
    axes[2].spines["left"].set_linewidth(0.5)
    axes[2].spines["right"].set_linewidth(0.5)
    axes[2].spines["top"].set_linewidth(0.5)
    axes[2].spines["bottom"].set_linewidth(0.5)
    axes[2].grid(linestyle="--", linewidth="0.5", color="grey", alpha=0.5)

    plt.grid(True)
    plt.tight_layout()
