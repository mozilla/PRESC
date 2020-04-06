import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
import matplotlib.pyplot as plt
import pandas as pd
from dataloader import get_x_y

def performance_over_folds(clf, num_fold=10):
    """
	Evaluates the performance of the model over different fold sizes.

	Parameters:
		clf : trained classification model
		num_fold : int

	Returns:
		fit_times: array-like
		train_scores: array-like
		test_scores: array-like
	"""

    folds = np.arange(2, num_fold + 1)
    fit_times = np.zeros(len(folds))
    train_scores = np.zeros(len(folds))
    test_scores = np.zeros(len(folds))

    x, y = get_x_y()

    for i, fold in enumerate(folds):
        fit_times[i] = np.mean(
            cross_validate(clf, x, y, cv=fold, return_train_score=True)["fit_time"]
        )
        train_scores[i] = np.mean(
            cross_validate(clf, x, y, cv=fold, return_train_score=True)["train_score"]
        )
        test_scores[i] = np.mean(
            cross_validate(clf, x, y, cv=fold, return_train_score=True)["test_score"]
        )

    return fit_times, train_scores, test_scores

def plot_fold_performance(clf, num_fold=10):
    """
	Visualises the Models performance over differnt fold numbers.

	Parameters:
		clf : trained classification model

	Returns:
		null
	"""

    fit_times, train_scores, test_scores = performance_over_folds(clf, num_fold)

    fig, axes = plt.subplots(3, 1)

    axes[0].plot(1000 * fit_times, color="xkcd:cherry", marker="o")
    axes[0].set_xlabel("Number of Folds")
    axes[0].set_xlim(left=2)
    axes[0].set_ylabel("Fit time(s)")
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

    axes[1].plot(100 * train_scores, color="xkcd:royal blue", marker="o")
    axes[1].set_xlabel("Number of Folds")
    axes[1].set_xlim(left=2)
    axes[1].set_ylabel("Train Score(%)")
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

    axes[2].plot(100 * test_scores, color="xkcd:emerald", marker="o")
    axes[2].set_xlabel("Number of Folds")
    axes[2].set_xlim(left=2)
    axes[2].set_ylabel("Test Scores (%)")
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
