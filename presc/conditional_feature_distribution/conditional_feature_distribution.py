import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_histogram_conditional_feature_distribution(
    y_predict, feature_column, y_actual, bins: int = None
):
    """
    Description:
        Plots histograms that represens the distribution of a feature.
        Each histogram should corresponds to a group on the confusion matrix of the test data.

    Output:
        X-axis represents the bins over the range of column values for that feature.
        Y-axis represents raw frequency counts.
        If categorical, then the x-axis would be categories instead of bins.

    Args:
        y_predict (pd.DataFrame): Predicted labels for a test set from a trained model
        feature_column (object): Column of data from the test set
        y_actual (pd.DataFrame): True labels for a test set from a trained model
        bins (int): Number of bins for the histograms. Defaults to None
    """
    # TODO: plot the histograms all on the same page in the order as the cofusion matrix
    # check if the feature_column is either a pandas series of list
    if not isinstance(feature_column, (pd.Series, list)):
        raise (
            TypeError(
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
    y_a = y_actual.astype("str")
    y_p = y_predict.astype("str")
    confusion_matrix_group = y_a.str.cat(y_p, sep="_predicted_as_")

    # make a dataframe for feature column and confusion matrix group
    feature_name = "feature"
    if isinstance(feature_column, pd.Series):
        feature_name = feature_column.name

    histo_df = pd.DataFrame(
        {feature_name: feature_column, "confusion_matrix_group": confusion_matrix_group}
    )

    # plot histogram for each unique group
    group_sizes = []
    for group, group_df in histo_df.groupby("confusion_matrix_group"):
        group_df = histo_df.loc[histo_df["confusion_matrix_group"] == group][
            feature_name
        ]
        group_sizes.append(group_df.size)
        plot_histogram(group, feature_name, group_df, bins)

    # purely returning this for testing purposes
    return group_sizes


def plot_histogram(
    confusion_matrix_group_name: str,
    feature_name: str,
    group_df,
    bins: int = None,
):
    """
    Description:
        Plots a single histogram that corresponds to a quadrant on the confusion matrix

    Args:
        confusion_matrix_group_name (str): The group that corresponds to a quadrant on the confusion matrix plot
        group_df (pd.DataFrame): the data that contains the feature values for a group
        bins (int, optional): Use Freedman Diaconis's methods to get bin numbers if not specified. Defaults to None.
    """

    # check whether the feature column is categorical of numeric
    if group_df.dtypes in (int, float):
        feature_array = group_df.to_numpy()

        # calcluate for a defualt bin number using Freedman–Diaconis rule
        if bins is None:
            bins = freedman_diaconis(group_df)

        plt.hist(x=feature_array, bins=bins)

        plt.xlabel(group_df.name)
        plt.ylabel("Frequency")
        plt.title("Group: " + confusion_matrix_group_name)
    else:
        x = np.arange(group_df.value_counts().size)
        plt.bar(height=group_df.value_counts(), x=x)

        plt.xlabel(group_df.name)
        cateogries = group_df.unique()
        plt.xticks(x, cateogries)
        plt.ylabel("Frequency")
        plt.title("Group: " + confusion_matrix_group_name)

    plt.show(block=False)


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
