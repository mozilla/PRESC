import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression  # import model to be used

def traversal(estimator, data):
    """ This function takes estimator and dataset as parameters
    and runs training on an estimator with different K-Fold 
    values. And it returns the result in tabular format.
    
    Args:
        estimator: Model to train
        data: Dataset
    Returns:
        results: A table of results with K-Fold values
        and Average Metric Value in each row 
    """

    results = list()

    features = data.drop(["Class"], axis=1)
    target = data.Class
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.4, random_state=42
    )

    # fit the training data into the model
    estimator.fit(train_features, train_target)

    # store the predicted values of test data
    predict_target = estimator.predict(test_features)

    # Evaluating the average metric value with a grid of K values from 1 to n in cross validation
    for split in range(2, 21, 1):
        score = cross_val_score(
            estimator, features, target, cv=split, scoring="accuracy"
        )
        mean_score = np.mean(score)  # calculate the mean of scores
        mean_score = round(mean_score, 5)  # round to 5 decimal places
        results.append([split, mean_score])

    # Convert the list to dataframe
    table = pd.DataFrame(results, columns=["Number of Folds", "Average Metric Value"])
    return table
