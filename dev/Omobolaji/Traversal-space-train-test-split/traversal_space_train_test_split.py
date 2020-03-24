import pandas as pd
import sklearn
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

    results = list()

    features = data.drop(["Class"], axis=1)
    target = data.Class

    for split in range(10, 100, 10):  # range(start, stop, step)
        split = split / 100  # convert to decimal format
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

        # Compute confusion matrix
        cm = confusion_matrix(test_target, predict_target)

        # Calculate the classification report
        cr = classification_report(test_target, predict_target)

        # append to an empty list created
        results.append([split, accuracy, cm, cr])

    # convert the results to tabular form using Pandas Dataframe function
    results = pd.DataFrame(
        results,
        columns=[
            "Test-Size",
            "Accuracy Score(%)",
            "Confusion Matrix",
            "Classification Report",
        ],
    )
    #     results.explode('Classification Report')   this function doesnt seem to work
    return results
