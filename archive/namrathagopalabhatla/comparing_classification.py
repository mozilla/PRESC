import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_processing import convert_binary
from classification_models import Classifier


def compare(models=None, X_train=None, y_train=None, X_test=None, y_test=None, group=[(0, 3), (1, 2)], labels=[0, 1]):
    """Function to plot the percentage of samples of each unique 
    predicted confidence/probability for a number of binary classifiers"""
    if models is None:
        print('Models not specified.')
        return
    elif 'mlp' in models:
        print('MLP not supported.')
        return
    elif X_train is None or y_train is None or X_test is None or y_test is None:
        print('Data not provided.')
        return

    allowed_models = ['logistic', 'svm', 'knn']
    
    if len(models) < 2:
        models = allowed_models

    y_train = convert_binary(y_train, group=group, labels=labels)
    y_test = convert_binary(y_test, group=group, labels=labels)

    if y_train is None or y_test is None:
        return

    probabilities = {}

    for model in models:
        classifier = Classifier(model)
        classifier.train(X_train, y_train)
        classifier.validate(X_test, y_test)
        predictions = classifier.classify(X_test)
        combined_probabilities = classifier.prediction_probabilities()
        indices = (np.arange(len(predictions)).astype(int), predictions.astype(int))
        probabilities[model] = combined_probabilities[indices]
        print('Model: {0}, Accuracy: {1}'.format(model, classifier.model_accuracy()))

    plt.figure(figsize=(15, 10))
    plt.hist(list(probabilities.values()), label=models, weights=[np.ones(len(y_test)) / len(y_test) for i in range(len(models))])
    plt.xlabel('Probability')
    plt.ylabel('Percentage of Data Samples')
    plt.legend()
    plt.show()

