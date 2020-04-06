from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# inputs:  estimator, X, y, cv, scoring
def train_test_split_table_scores(estimator, X, y, cv):
    """ returns table that shows train and test split percentages as well as per split metric with average over train and test scores """
    #     train_percent_float = np.array(list(range(1, 101))) / 100

    #     train_sizes = (train_percent_float*int(len(X) - len(X)*0.2)).astype(int)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator, X=X, y=y, cv=cv, shuffle=True
    )

    #     train_percent = (train_percent_float*100).astype(int)
    #     test_percent =100 -print('Training scores:\n\n', train_scores) train_percent

    train_percent = np.round(train_sizes * 100 / len(X), 2)
    test_percent = (len(X) - train_sizes) * 100 / len(X)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    column_names = [("Split" + str(i)) for i in range(1, cv + 1)]

    table_of_train_test_split = pd.concat(
        [
            pd.DataFrame(
                {"Train Percent": train_percent, "Test Percent": test_percent}
            ),
            pd.DataFrame(train_scores, columns=column_names),
            pd.DataFrame(
                {
                    "Training Scores Mean": train_scores_mean,
                    "Testing Scores Mean": validation_scores_mean,
                }
            ),
        ],
        axis=1,
    )

    return table_of_train_test_split


def visual_tain_test_split_score(table):
    """ visualises table of train data percent Vs avg train scores and train data percent Vs avg test score. """
    plt.style.use("seaborn")
    plt.plot(
        table["Train Percent"], table["Training Scores Mean"], label="Training Score"
    )
    plt.plot(
        table["Train Percent"], table["Testing Scores Mean"], label="Validation Score"
    )
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.title("Learning curves", fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0.5, 1)
