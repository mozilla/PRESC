import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def load_split_preprocessed_data(i):
    """

        This function takes in i, which defines by what fraction we want to undersample our data
        It returns processed data, which is split into test-train, is undersampled, has only the features that we want and is normalised to take care of the skewness

    """
    data_path = "../../../datasets/winequality.csv"
    data = pd.read_csv(data_path)
    count_false_recommendation = len(data["recommend"] == True)
    true_indices = np.array((data.recommend == True).index)
    false_indices = np.array((data.recommend == False).index)

    undersample_data = undersample(
        true_indices, false_indices, i, count_false_recommendation, data
    )

    # since data is skewed, we add epsilon to data and take its logarithm
    epsilon = 0.00000001

    undersample_data[
        [
            "fixed acidity",
            "volatile acidity",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "sulphates",
            "alcohol",
            "quality",
        ]
    ] = np.add(
        undersample_data[
            [
                "fixed acidity",
                "volatile acidity",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "sulphates",
                "alcohol",
                "quality",
            ]
        ],
        epsilon,
    )
    undersample_data[
        [
            "fixed acidity",
            "volatile acidity",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "quality",
        ]
    ] = np.log(
        undersample_data[
            [
                "fixed acidity",
                "volatile acidity",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
                "quality",
            ]
        ]
    )
    undersample_data.drop(["quality"], axis=1, inplace=True)
    x_features = undersample_data.drop(["recommend"], axis=1, inplace=False)
    x_labels = undersample_data["recommend"]

    # splitting data on the basis of accuracy with different sets. Accuracy didn't vary much with the split, but gave highest with this number. Hence, you are free to go ahead with any number.

    test_size = 0.3
    x_features_train, x_features_test, x_labels_train, x_labels_test = train_test_split(
        x_features, x_labels, test_size=test_size
    )

    return (x_features_train, x_features_test, x_labels_train, x_labels_test)


# undersampling data
def undersample(true_indices, false_indices, times, count_false_recommendation, data):
    """

    This function takes in true indices (recommended) and false indices (not recommended) data points and times, which is the fraction of data points we will have of the total unrecommended data points, ie undersampling the data.
    It returns undersampled data, with recommended data points as it is, and unrecommended data points undersampled.

    """
    false_indices_undersample = np.array(
        np.random.choice(
            false_indices, (times * count_false_recommendation), replace=True
        )
    )
    undersample_data = np.concatenate([true_indices, false_indices_undersample])
    undersample_data = data.iloc[undersample_data, :]
    return undersample_data
