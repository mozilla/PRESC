import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from classification_models import Classifier
from sklearn.model_selection import KFold

from IPython.display import display


def cross_validation_variation(X=None, y=None, limits=(2, 15, 1)):
    """Function used to perform K-fold validation on the provided data to do a comparison of
    the number of splits and the corresponding change in accuracies of prediction"""
    if X is None or y is None:
        print("Data not provided.")
        return
    elif limits[0] > limits[1]:
        print("Invalid limits.")
        return

    splits = []
    mean_accuracies = []
    for split in range(max(2, limits[0]), limits[1], max(1, limits[2])):
        k_fold = KFold(n_splits=split)
        splits.append(split)

        accuracies = []

        classifier = Classifier(model="logistic")
        for train, test in k_fold.split(X):
            classifier.train(X[train, :], y[train])
            classifier.validate(X[test, :], y[test])
            accuracies.append(classifier.model_accuracy())

        mean_accuracies.append(np.mean(np.array(accuracies)))

    table = pd.DataFrame({"Splits": splits, "Mean Accuracy": mean_accuracies})
    display(table)

    plt.figure(figsize=(10, 10))
    plt.plot(splits, mean_accuracies, "-o")
    plt.xlabel("Number of Splits")
    plt.ylabel("Mean Accuracy")
    plt.show()
