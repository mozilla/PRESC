from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
import pandas as pd


def sensitivity(clf, X, y, accuracy):

    """This function loop through total numbers of rows. In each loop, it removes that data point and 
    trains the classifier and calculates the accuracy for that pass. The calculated accuracies for each pass
    is stored in the list.
    
    
    Parameters
    ----------
    clf: Classifier
    X : Data with Feature Column.
    y : Label for each data row.
    accuracy: Base Accuracy for the dataset.
    
    Returns
    -------
    datasets_df : Dataframe with the list of all the accuracy details
    """

    datasets_list = list()
    for i in range(0, (X.shape[0] - 1)):
        r_class = y.iloc[i][0]

        X_new = X.iloc[X.index != i]
        y_new = y.iloc[y.index != i]

        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new.values.ravel(), test_size=0.3, random_state=1
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        n_accuracy = round(accuracy_score(y_test, y_pred), 2)
        diff_in_accuracy = abs(n_accuracy - accuracy)
        datasets_list.append(
            [
                i,
                r_class,
                accuracy,
                n_accuracy,
                (n_accuracy - accuracy),
                diff_in_accuracy,
            ]
        )

    datasets_df = pd.DataFrame(
        datasets_list,
        columns=[
            "Idx of Removed Row",
            "Class",
            "Base Accuracy",
            "Accuracy",
            "Change in Accuracy",
            "Change in Accuracy(ABS)",
        ],
    )
    return datasets_df
