# Import necessary modules
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import k_nn


def vary_size(X, y):
    """This function trains tests and evaluates the performance of a dataset in comparison
    with a changing train-to-test size ratio. It returns two lists of values; accuracy and test_ratio which
    containing the values of accuracy and test_size respectively"""
    accuracy = []
    test_ratio = []
    performance = "\n PERFORMANCE \n "
    """starting the range iterator from zero  or ending at 101 raises a ValueError: 
    The test_size = 0 (1.0) should be greater or equal to the number of classes = 4"""
    for i in range(5, 100, 5):
        size = i / 100
        # split data set into train and test sets
        data_train, data_test, target_train, target_test = train_test_split(
            X, y, test_size=size, random_state=10, stratify=y
        )
        # Evaluation of the  performance of the K-nearest neighbors prediction model
        kn_accuracy, target_pred = k_nn.k_nearest(
            data_train, target_train, data_test, target_test
        )
        # generate classification report to observe values of precision, recall, f1_score and support
        class_report = classification_report(target_test, target_pred)
        # separator string demarcates results of one iteration from the other
        separator = ("+" * 100) + " \n"
        performance = performance + separator + class_report + " \n"
        # update lists
        accuracy.append(kn_accuracy)
        test_ratio.append(size)

    return (test_ratio, accuracy, performance)
