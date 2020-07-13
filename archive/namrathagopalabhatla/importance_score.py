import matplotlib.pyplot as plt
import numpy as np

from data_processing import split_data, dim_reduction


def importance_score(model=None, X_train=None, y_train=None, X_test=None, y_test=None, metric='accuracy'):
    '''
    Function to check the importance of each training data point computed as the difference between the 
    metric computed by using all points and that computed leaving the point out. The function generates 3 plots:
    1. Plot of 10 most and least important data points
    2. Plot of data point index vs weights
    3. Plot of all data points color-coded by weight
    '''
    if model is None:
        print("Model not provided.")
        return
    
    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("All the required data is not provided.")
        return

    baseline = None
    current = None
    weights = []

    reduced_X = dim_reduction(method='pca', X=X_train, k=2)

    model.train(X_train, y_train)
    model.validate(X_test, y_test)
    
    if metric == 'accuracy':
        baseline = model.model_accuracy()
    elif metric == 'score':
        baseline = model.model_score()

    for i in range(X_train.shape[0]):
        X = np.delete(np.copy(X_train), i, axis=0)
        y = np.delete(np.copy(y_train), i, axis=0)
        
        model.train(X, y)
        model.validate(X_test, y_test)

        if metric == 'accuracy':
            current = model.model_accuracy()
        elif metric == 'score':
            current = model.model_score()

        weights.append(baseline - current)

    indices = np.argsort(np.array(weights))

    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_X[indices[:10], 0], reduced_X[indices[:10], 1], c='r', label='10 Least important points')
    plt.scatter(reduced_X[indices[-10:], 0], reduced_X[indices[-10:], 1], c='g', label='10 Most important points')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Most & Least Important Points')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(X_train.shape[0]), weights)
    plt.xlabel('Datapoint index')
    plt.ylabel('Weight')
    plt.title('Index vs Weight')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=weights, cmap='coolwarm')
    plt.colorbar(shrink=0.5, aspect=5)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Training data points color-coded as per Weights')
    plt.show()