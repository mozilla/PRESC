import pandas as pd
import allcustommodules as sidra
import matplotlib.pyplot as plt
import numpy as np


def missclassified_data_category_frquency(y_test, y_pred):
    """ This function plots frequency of missclassified points against the incorrect categories they were predicted for
    parameters: 
    y_test: reverse factorized test values
    y_pred: reverse factorized predicted values """
    cm = sidra.create_confusion_matrix(y_test, y_pred)
    for label in cm.columns:
        cm.at[label, label] = 0

    ax = cm.plot(kind="bar", title="Predicted Class Error", stacked=True)
    ax.locator_params(axis="y", integer=True)
    ax.set_xlabel("Classes/Categories")
    ax.set_ylabel("Number of Incorrectly Predicted Class")
    plt.show()


def untokenize_test_predict_data(definition, y_test, y_pred):
    """ this function can be used to reverse factor test and predicted values before using 'missclassified_data_category_frquency' function
    parameters: 
    y_test:  factorized test values
    y_pred: factorized predicted values
    definitions: categories for reverse factorizing"""
    reversefactor = dict(zip(range(len(definitions) + 1), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
