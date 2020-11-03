"""
We compute the distribution of a feature over the test set restricted to each cell of the confusion matrix.
This allows us to compare distributions between misclassified and correctly classified points.

Input [pandas series or list]:

Predicted labels for a test set from a trained model
Column of data from the test set (eg. values of a feature)
could also be predicted scores or a function of the features


Output:
Histograms of distribution of feature in each confusion matrix
"""
import pandas as pd
import matplotlib.pyplot as plt

# TODO: What about the case where there is a class that does not exist in either ys?


def plot_all_histogram_conditional_feature_distribution(
    y_predict, feature_column, y_actual, bins: int = None
):
    # TODO: fix the description
    """
    Description:
        Plots one histogram that represens the distribution of a feature.

        Input:
            A pandas series or list that is the same length as the dataset

        Output:
            X-axis represents the bins over the range of column values for that feature.
            Y-axis represents relative frequency counts.
            If categorical, then the x-axis would be categories instead of bins.

    Args:
        predicted_labels ([str]): [Predicted labels for a test set from a trained model]
        feature_column (object): [Column of data from the test set]
        bins (int): [Number of bins for the histograms. Use matplotlib default if unspecified]
    """

    # check if the feature_column is either a pandas series of list
    if not isinstance(feature_column, (pd.Series.dtype, list)):
        raise (
            ValueError(
                "The feature column should be either a list or a pandas Series object"
            )
        )

    # check if feature_column is the same length as the dataset length
    if len(feature_column) != len(y_actual) or len(y_predict) != (y_actual):
        raise (
            ValueError(
                "The feature column should have the same length as the dataset and the predicted labels"
            )
        )

    # create confusion matrix label for each row
    confusion_matrix_group = []

    for i in range(len(y_actual)):
        label = str(y_actual[i]) + "_predicted_as" + str(y_predict[i])
        confusion_matrix_group[i] = label

    # make a dataframe for feature column and confusion matrix group
    feature_name = "feature"
    if isinstance(feature_column, pd.Series.dtype):
        feature_name = feature_column.name

    histo_df = pd.DataFrame(
        {feature_name: feature_column, "confusion_matrix_group": confusion_matrix_group}
    )

    # plot histogram for each unique group
    unique_groups = sorted(list(set(confusion_matrix_group)))

    for group in unique_groups:
        group_df = histo_df.loc[
            histo_df["confusion_matrix_group"] == group
        ].feature_name
        plot_histogram(group, group_df, bins)


def plot_histogram(
    confusion_matrix_group_name, group_specific_feature_column, bins: int = None
):

    # check whether the feature column is categorical of numeric
    if group_specific_feature_column.dtype in (int, float):
        feature_array = group_specific_feature_column.to_numpy()

        plt.his(x=feature_array, bins=bins)

        plt.xlabel(group_specific_feature_column.name)
        plt.ylabel("Frequency")
    else:
        group_specific_feature_column.value_counts().plot(kind="bar")

    plt.show()
