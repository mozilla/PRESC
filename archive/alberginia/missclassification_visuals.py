import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


def predictions_to_class(X_test, y_test, y_predicted, new_classes="hits-fails"):
    """Converts the data location on the confusion matrix into classes.

    This function builds a dataset where the classification of the data
    points is related to the confusion matrix, allowing to analyse
    visually in depth the missclassifications. Depending on the
    selected option for the parameter "new_classes", the classes
    reflect different aspects of the confusion matrix.

    Parameters:
        X_test (DataFrame): Subset of the original data without the
            class to use for validation of the model.
        y_test (Series): Class corresponding to the X_test subset to
            use for validation of the model.
        y_predicted (array): List of predictions yielded by the model
            after being trained corresponding to the X_test dataset.
        new_classes (str): Different options for the class output
            depending on the desired analysis.
            * "hits-fails" -> Allows to visualize the overall
              distributions of correctly classified points vs
              missclassified points (2 classes: correctly classified
              points and missclassified points).
            * "which-hit" -> Allows to visualize which combinations of
              values make each class to be missclassified (original
              number of classes + 1: missclassified points, and original
              classes for the correctly classified points).
            * "which-fail" -> Allows to visualize which combination of
              values make each class to be correctly classified
              (original number of classes + 1: correctly classified
              points + original classes for the missclassified points).

    Returns:
        dataset_missclass (DataFrame): original dataset but with a new
            set of classes, depending on the selected option.
    """
    dataset_missclass = X_test.copy()
    dataset_missclass_calc = X_test.copy()
    dataset_missclass_calc["Class"] = y_test
    dataset_missclass_calc["Predicted class"] = y_predicted
    dataset_missclass_calc["Hit"] = (
        dataset_missclass_calc["Class"] == dataset_missclass_calc["Predicted class"]
    )

    if new_classes == "hits-fails":
        dataset_missclass["Miss & Class"] = dataset_missclass_calc["Hit"].map(
            {True: "> Prediction hit", False: "> Prediction fail"}
        )

    elif new_classes == "which-hit":
        dataset_missclass["Miss & Class"] = np.where(
            dataset_missclass_calc["Hit"] is False,
            dataset_missclass_calc["Hit"].replace(False, "> Prediction fail"),
            dataset_missclass_calc["Class"].astype(str),
        )

    elif new_classes == "which-fail":
        dataset_missclass["Miss & Class"] = np.where(
            dataset_missclass_calc["Hit"] is True,
            dataset_missclass_calc["Hit"].replace(True, "> Prediction hit"),
            dataset_missclass_calc["Class"].astype(str),
        )

    return dataset_missclass
