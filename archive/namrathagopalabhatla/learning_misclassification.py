import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from classification_models import Classifier
from data_processing import convert_binary


def get_misclassifications(predictions=None, truth=None):
    if predictions is None or truth is None:
        return None
    else:
        misclassified = np.where(predictions != truth)[0]
        return misclassified


def learn_from_misclassified(model=None, X=None, y=None):
    """
    Function that generates 6 plots: 
    - Prediction probability vs Sample index (Misclassified points only)
    - Prediction probability vs Sample index (Correctly classified points only)
    - 4 plots (for each class):
    Distance of misclassified from class mean (with reference line for mean of distance of correctly classified)
    """
    if model is None or X is None or y is None:
        print("All required parameter not provided.")
        return

    predictions = model.classify(X)
    misclassified = get_misclassifications(predictions, y)
    correct = np.delete(np.arange(y.shape[0]), misclassified)

    model.validate(X, y)
    probabilities = model.prediction_probabilities()
    mis_probabilities = probabilities[misclassified]
    correct_probabilities = probabilities[correct]

    print("Predicted Probability vs Misclassified Sample Indices")
    print("Note: Legend colour indicates the true class of the point")
    plt.figure(figsize=(10, 10))

    for i in np.unique(y):
        plt.subplot(2, 2, i + 1)
        for j in np.unique(y):
            plt.scatter(
                misclassified[y[misclassified] == j],
                mis_probabilities[:, i][y[misclassified] == j],
                label="Class {}".format(j),
            )
        plt.title("Class {}".format(i))
        plt.ylabel("Prediction Probability")
        plt.xlabel("Misclassified Sample Indices")
        plt.legend()

    plt.show()

    print("Predicted Probability vs Correctly Classified Sample Indices")
    print("Note: Legend colour indicates the true class of the point")
    plt.figure(figsize=(10, 10))

    for i in range(probabilities.shape[1]):
        plt.subplot(2, 2, i + 1)
        for j in np.unique(y):
            plt.scatter(
                correct[y[correct] == j],
                correct_probabilities[:, i][y[correct] == j],
                label="Class {}".format(j),
            )
        plt.title("Class {}".format(i))
        plt.ylabel("Prediction Probability")
        plt.xlabel("Correctly Classified Sample Indices")
        plt.legend()

    plt.show()

    classes = {}  # class-wise indices of samples
    means = {}
    miss = {}  # indices of misses
    hits = {}  # indices of hits
    reference = {}
    miss_distances = {}

    for i in np.unique(y):
        classes[i] = np.where(y == i)[0]
        means[i] = np.mean(X[classes[i], :], axis=0)

        indices = get_misclassifications(predictions[classes[i]], y[classes[i]])
        miss[i] = classes[i][indices]
        hits[i] = np.delete(np.copy(classes[i]), indices)

        reference[i] = np.mean(
            np.linalg.norm((X[hits[i], :] - means[i]).astype(float), axis=1)
        )
        miss_distances[i] = np.linalg.norm(
            (X[miss[i], :] - means[i]).astype(float), axis=1
        )

    for i in np.unique(y):
        plt.figure(figsize=(10, 8))
        plt.plot(
            np.arange(miss[i].shape[0]),
            miss_distances[i],
            marker="o",
            label="Misclassified Samples",
        )
        plt.plot(
            np.arange(miss[i].shape[0] + 2),
            np.ones(miss[i].shape[0] + 2) * reference[i],
            linestyle="--",
            label="Reference",
        )
        plt.xlabel("Sample index (in class)")
        plt.ylabel("Distance from Class Mean")
        plt.title(
            "Distance of Misclassified Samples from Class Mean for Class {}".format(i)
        )
        plt.show()
