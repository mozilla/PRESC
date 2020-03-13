from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def tune(data_train, target_train):
    """This function is for hyperparameter tuning in order to know the most appropriate value of 
    n_neighbors to use in the k_nearest() function. """
    # create hyperparameter grid
    c_space = np.arange(1, 9)
    # the value of n_neighbors has to be an integer and greater than zero.
    params_grid = {"n_neighbors": c_space}
    # instantiate the classifier
    kn = KNeighborsClassifier()
    # Instantiate GridSearchCV object
    kn_cv = GridSearchCV(kn, params_grid, cv=5, iid=True)

    """Increasing the number of folds increases the accuracy of the model performance evaluation index.
    However, this tends to be very computationally expensive in terms of runtime and memory usage.
    Five folds is just a compromise default value """
    kn_cv.fit(data_train, target_train)
    """Get the value corresponding to the n_neighbors key in the dictionary returned by best_params_ . 
    This value corresponds to the n_neighbor value that produced the best estimator performance. """
    param_val = kn_cv.best_params_["n_neighbors"]
    return param_val


def k_nearest(data_train, target_train, data_test, target_test):
    """Training model using the k nearest neighbor algorithm. This function trains 
    the model using a tuned hyperparameter value. In it, the accuracy of 
    the K-nearest neighbor algorithm on this data set is computed and returned. 
    note that 'neighbor' must be spelt without including a 'u' in order for it to work """

    n = int(tune(data_train, target_train))
    """ The param_val value is converted from the numpy.int64 type to an integer in order to make it appropriate
    for use as a parameter of the classifier."""
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(data_train, target_train)

    # predict labels of the test data
    target_pred = knn.predict(data_test.values)

    # The held out test set is used to evaluate the accuracy of the model
    acc = knn.score(data_test, target_test)
    return (acc, target_pred)


def with_without(df, df_without, targetname):
    """This function computes the accuracy of a KNN model over different configurations of its
    training dataset. It returns a list containing the accuracy values per dataset configuration. """
    lst = []
    for df in [df, df_without]:
        data = df.drop(targetname, axis=1)
        target = df[targetname]
        # axis=1 indicates that the string is a column not a row
        data_train, data_test, target_train, target_test = train_test_split(
            data, target, test_size=0.30, random_state=10, stratify=target
        )
        kn_accuracy = k_nearest(data_train, target_train, data_test, target_test)
        lst.append(kn_accuracy)
    return lst


def visual_compare(data_train, target_train, data_test, target_test):
    """This function outputs a graph of the accuracy of the test set 
    compared to that of the train set as gotten from the K-NN algorithm. 
    The variation between the train accuracy (reference) and test accuracy 
    is clearly visualised such that the user can choose the optimal value of 
    the n_neighbors parameter of KNeighborsClassifier that will yield maximum
    prediction accuracy."""

    neighbors = np.arange(1, 9)
    """The values chosen here are arbitrary. Increasing the range only increased the
    run time of the function and no visible changes were seen in the pattern of the graph.
    In the case of a less ideal dataset, there is a possibility that increasing the values in
    np.arange() will give the user a larger scope to observe if there are other values of n_neighbors
    that will yield a better prediction accuracy. Arrays are setup to store train and test accuracies
    as shown below; """
    train_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(data_train, target_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(data_train, target_train)

    # Compute accuracy on the testing set using cross validation

    test_accuracy = cross_val_score(knn, data_test, target_test, cv=len(neighbors))

    # Generate plot
    plt.title("k-NN: Varying Number of Neighbors")
    plt.scatter(neighbors, test_accuracy, label="Testing Accuracy", color="red")
    """.plot is used for the train accuracy demonstration because its points were observed 
    to superimpose themselves on the plot points of test accuracy. Using .plot() for one 
    and .scatter() for the other increases visibility"""
    plt.plot(neighbors, train_accuracy, label="Training Accuracy", color="blue")
    plt.legend()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()
