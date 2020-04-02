import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_processing import split_data
from classification_models import Classifier


def split_variation(X=None, y=None, step=0.05):
    """ Function used to split the data into different ratios for the training and the testing data where,
         the training data varies in "step" intervals as specified and the LR classifier is trained
        with the given splits and then we obtain the respective accuracy and F1-Score"""
    if X is None or y is None:
        print("Data not provided.")
        return

    split = 0.00
    splits = []
    accuracies = {"train": [], "test": []}
    f1_scores = {"train": [], "test": []}

    for split in np.arange(step, 1.0, step):
        splits.append(split * 100)
        X_train, X_test, y_train, y_test = split_data(X, y, split)

        classifier = Classifier(model="logistic")
        classifier.train(X_train, y_train)

        classifier.validate(X_train, y_train)
        train_accuracy = classifier.model_accuracy()
        train_score = classifier.model_score()

        classifier.validate(X_test, y_test)
        test_accuracy = classifier.model_accuracy()
        test_score = classifier.model_score()

        accuracies["train"].append(train_accuracy)
        accuracies["test"].append(test_accuracy)
        f1_scores["train"].append(train_score)
        f1_scores["test"].append(test_score)

    splits = np.array(splits)

    training = {
        "accuracy": np.array(accuracies["train"]),
        "score": np.array(f1_scores["train"]),
    }

    testing = {
        "accuracy": np.array(accuracies["test"]),
        "score": np.array(f1_scores["test"]),
    }

    table = pd.DataFrame(
        {
            "Train size (%)": splits,
            "Training Accuracy": training["accuracy"],
            "Training F1-Score": training["score"],
            "Testing Accuracy": testing["accuracy"],
            "Testing F1-Score": testing["score"],
        }
    )

    table.style.set_caption("Variation of performance with train-test split")
    display(table)

    plt.figure(figsize=(15, 10))
    plt.plot(splits, training["accuracy"], c="blue", label="Training Accuracy")
    plt.plot(splits, testing["accuracy"], c="green", label="Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Set Size (%)")
    plt.title("Training Set Size vs Accuracy")
    plt.legend()
    plt.show()
