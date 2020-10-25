"""
We compute the distribution of a feature over the test set restricted to each cell of the confusion matrix.
This allows us to compare distributions between misclassified and correctly classified points.

Input:

Predicted labels for a test set from a trained model
Column of data from the test set (eg. values of a feature)
could also be predicted scores or a function of the features

Output:
Distributional representation (eg. value counts, histogram or density estimate) for each cell in the confusion matrix
"""
import pandas as pd

# TODO: what to output if the input is "predicted scores or a function of the features" ?


def get_value_counts_conditional_feature_distribution(
    predicted_labels, test_data_column
):
    labels = list(set(test_data_column))

    y_actual = pd.Categorical(test_data_column, categories=labels)
    y_predicted = pd.Categorical(predicted_labels, categories=labels)

    confusion_matrix = pd.crosstab(
        y_actual, y_predicted, rownames=["Actual"], colnames=["Predicted"], dropna=False
    )

    return confusion_matrix


# TODO: re-use code to implement this?


def get_histogram_conditional_feature_distribution(
    predicted_labels: [str], feature_column: object, bins: int = None
):

    """
    Description:
        Outputs a set of the histograms for specified column. Each corresponds to one label/class.

    Args:
        predicted_labels ([str]): [Predicted labels for a test set from a trained model]
        feature_column (object): [Column of data from the test set]
        bins (int): [Number of bins for the histograms. Use matplotlib default if unspecified]
    """
    pass
