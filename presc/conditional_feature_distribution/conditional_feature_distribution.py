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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_histogram_conditional_feature_distribution(
    y_predict, feature_column, y_actual, bins: int = None
):
    # TODO: fix the description
    # TODO: deal with bins
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
    if not isinstance(feature_column, (pd.Series, list)):
        raise (
            ValueError(
                "The feature column should be either a list or a pandas Series object"
            )
        )

    # check if feature_column is the same length as the dataset length
    if len(feature_column) != y_actual.size or y_predict.size != y_actual.size:
        raise (
            ValueError(
                "The feature column should have the same length as the dataset and the predicted labels"
            )
        )

    # create confusion matrix label for each row
    confusion_matrix_group = []
    for i in range(y_actual.size):
        label = str(y_actual.iat[i]) + "_predicted_as_" + str(y_predict[i])
        confusion_matrix_group.append(label)

    # make a dataframe for feature column and confusion matrix group
    feature_name = "feature"
    if isinstance(feature_column, pd.Series):
        feature_name = feature_column.name

    histo_df = pd.DataFrame(
        {feature_name: feature_column, "confusion_matrix_group": confusion_matrix_group}
    )

    # plot histogram for each unique group
    unique_groups = sorted(list(set(confusion_matrix_group)))

    for group in unique_groups:
        group_df = histo_df.loc[histo_df["confusion_matrix_group"] == group]
        plot_histogram(group, feature_name, group_df, bins)


def plot_histogram(
    confusion_matrix_group_name,
    feature_name,
    group_df,
    bins: int = None,
):

    # check whether the feature column is categorical of numeric
    if group_df[feature_name].dtypes in (int, float):
        feature_array = group_df[feature_name].to_numpy()

        # calcluate for a defualt bin number using Freedman–Diaconis rule
        if bins is None:
            bins = freedman_diaconis(group_df[feature_name])

        plt.hist(x=feature_array, bins=bins)

        plt.xlabel(feature_name)
        plt.ylabel("Frequency")
        plt.title("Group: " + confusion_matrix_group_name)
    else:
        # TODO: use plt.hist() for this to keep the code consistent
        group_df.value_counts().plot(kind="bar")

    plt.show()


def freedman_diaconis(data):
    IQR = np.diff(np.quantile(data, q=[0.25, 0.75]))[0]
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    if bw == 0:
        return 10

    datmin, datmax = data.min(), data.max()
    datrng = datmax - datmin
    bin_nums = int(datrng / bw) + 1

    return bin_nums
