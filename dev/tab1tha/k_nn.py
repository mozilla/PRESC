from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


def k_nearest(data_train, target_train, data_test, target_test):
    """This function trains a k nearest neighbor model then uses it to predict target values of the test data. In it, the accuracy
    of the K-nearest neighbor algorithm on this data set is computed and returned. 
    Note that 'neighbor' must be spelt without including a 'u' in order for it to work. """

    knn = KNeighborsClassifier(n_neighbors=5)
    y = target_train
    X = data_train

    # Fit the classifier to the training data
    knn.fit(X, y)

    # predict labels of the test data
    target_pred = knn.predict(data_test.values)
    # compute accuracy and return it
    acc = knn.score(data_test, target_test)
    return (acc, target_pred)


def visual_compare(data_train, target_train, data_test, target_test):
    """This function outputs a graph of the predictions gotten from the K-nn algorithm. In that same graph, the 
    actual values are plotted. The variation between the predicted values and actual values is clearly visualised such
    that the user can choose the optimal value of the n_neighbors parameter of KNeighborsClassifier that will yied maximum
    prediction accuracy"""
    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(data_train, target_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(data_train, target_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn.score(data_test, target_test)

    # Generate plot
    plt.title("k-NN: Varying Number of Neighbors")
    plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
    plt.plot(neighbors, train_accuracy, label="Training Accuracy")
    plt.legend()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()
