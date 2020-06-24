from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


def distanceFromDecisionBoundary(clf, X_test):
    """
    Computes the distance from decision boundary for test points.
    Args: clf - classifer, X_test - testing data
    Returns: distance_to_decision_boundary - relative distance from the decision boundary of test points.
    """
    distance_to_decision_boundary = clf.decision_function(X_test)
    return distance_to_decision_boundary


def plotDistancesFromDecisionBoundary(param, preds, y_test, param_name):
    """
    Plots the correctly classified(red) and misclassified(black) points w.r.t the decision boundary
    Args: param - distance computed from clf.decision_function or probabilities from predicted_proba, preds - predictions for X_test,
          y - label, param_name - string of param name
    """
    correct = [
        pred == true for pred, true in zip(preds, y_test)
    ]  # Which data points are misclassified
    misclassified = []  # misclassified indices
    param_mis = (
        []
    )  # distances or probabilities of misclassified data points from decision boundary
    correct_data = []  # correctly classifed indices
    param_cor = (
        []
    )  # distances or probabilities of correctly classifed data points from decision boundary
    i = 0
    for b, dist in zip(correct, param):
        if b == 0:
            misclassified.append(i)
            param_mis.append(dist)
        else:
            correct_data.append(i)
            param_cor.append(dist)
        i = i + 1
    print(len(misclassified))
    print(len(correct_data))
    fig = plt.figure(figsize=(15, 10))
    plt.plot(
        correct_data,
        param_cor,
        "o",
        color="red",
        alpha=0.5,
        ms=10,
        label="correctly classified",
    )  # correctly classified
    plt.plot(
        misclassified,
        param_mis,
        "o",
        color="black",
        alpha=0.5,
        ms=10,
        label="misclassified data",
    )  # misclassified data
    # plt.ax.set(option='auto')
    # plt.autoscale(enable=True, axis='x')#plt.axis('tight')
    plt.xlabel("Indices of all data")
    plt.ylabel(param_name)
    plt.legend()
    plt.xlim(0, 10000)
    plt.show()
