import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression

def traversal(estimator, data):
    """ This function takes estimator and a dataset as params and runs 
    training on estimator with different test sizes. Returns result of
    the test sizes and accuracy score in a tabular form.
    
    Args:
        estimator: Model to train
        data: Dataset
    Returns:
        results: A table of results with different test 
                sizes and their accuracy score. 
    """
    #     metrics = " \n "
    results = list()

    features = data.drop(["Class"], axis=1)
    target = data.Class

    for split in range(10, 100, 10):  # range(start, stop, step)
        split = split / 100  # convert to decimal format
        train_size = 1 - split
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=split, random_state=42
        )

        # fit the training data into the model
        estimator.fit(train_features, train_target)

        # store the predicted values of test data
        predict_target = estimator.predict(test_features)

        # Evaluate accuracy score
        accuracy = accuracy_score(test_target, predict_target)

        # convert the score to a percentage and round to two decimal places
        accuracy = round(accuracy * 100, 2)

        # Evaluate the weighted(average) precision, recall and F1 score
        """
        In calculating these scores, the parameter,
        "average" can be equated to micro(the smallest value), macro(the largest value) 
        or weighted(which is an average of the values). In order to have a good overall
        view of our scores for each test sizes, "weighted" is chosen.
        """

        precision = precision_score(test_target, predict_target, average="weighted")
        precision = round(precision * 100, 2)
        recall = recall_score(test_target, predict_target, average="weighted")
        recall = round(recall * 100, 2)
        f1 = f1_score(test_target, predict_target, average="weighted")
        f1 = round(f1 * 100, 2)

        # append to an empty list created
        results.append([train_size, split, accuracy, precision, recall, f1])

    # convert the results to tabular form using Pandas Dataframe function
    results = pd.DataFrame(
        results,
        columns=[
            "Train Size",
            "Test Size",
            "Accuracy Score(%)",
            "Precision(%)",
            "Recall(%)",
            "F1_Score(%)",
        ],
    )
    return results
