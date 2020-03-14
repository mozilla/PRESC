from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

def distanceFromDecisionBoundary(clf, X_test):
    '''
    Computes the distance from decision boundary for test points.
    Args: clf - classifer, X_test - testing data
    Returns: distance_to_decision_boundary - relative distance from the decision boundary of test points.
    '''
    distance_to_decision_boundary = clf.decision_function(X_test)
    return distance_to_decision_boundary
    