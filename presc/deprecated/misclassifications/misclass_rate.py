import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score


def misclass_rate_feature(
    test_dataset,
    test_predictions,
    feature,
    categorical=False,
    bins=10,
    bins_type="regular",
):
    """Computes the misclassification rate as a function of the feature values.
    This function allows to compute the misclassification rate of a sample for
    the values of any particular feature.
    The function allows any binning for this calculation, which means that
    regularly spaced binnings, disparately spaced binnings that correspond to
    sets of an equal amount of data points (such as quartiles, deciles, or
    n-quantiles), or any other arbitrary irregular binning can be used.
    When the full dataset with all points does not have any data point in an
    interval corresponding to a certain bin, the function yields a "nan" value
    for the misclassification rate to prevent a zero division error and also to
    distinguish the bins without information from the bins with a zero
    misclassification rate. The same happens with the standard deviation when
    either the full dataset with all points or the dataset with only the
    misclassified points do not have any data point in a certain bin interval.
    Parameters:
        test_dataset: Dataset with the features of all data points, where the
            true class is at the last column.
        test_predictions: List of the predicted classes for all data points.
        feature: Column name in the dataset of the feature.
        categorical (bool): Indicates whether it is a categorical feature.
            Default is "False".
        bins (int, list, str):
            * If an integer, this indicates the number of bins (default value is
            10). Whether this corresponds to dividing the feature scale in
            regularly spaced bins (default) or into quantiles, it must be
            specified in the parameter "bins_type".
            * If the string "quartiles" or "deciles" is used, then it
            automatically computes the appropriate bin edge postions to optimize
            for a quartile or decile grouping.
            * If any other feature intervals are needed, then a list of the
            feature values corresponding to the positions separating the bins
            and including the outermost edges must be provided.
        bins_type (str): If the bins parameter is an integer with the number
            of bins, this parameter allows to specify whether these bins should
            be "regular" evenly spaced bins or "quantiles". Default value is
            "regular".
    Returns:
        Three elements which correspond to:
            1) The edges of the bins in the feature scale.
            2) The misclassification rate in each bin.
            3) The standard deviation of the misclassification rate in that bin.
    """

    # metric_func computes [total count in each group, misclass count in each group]

    bins, metric_list = compute_conditional_metric(
        test_dataset,
        test_predictions,
        feature,
        categorical=categorical,
        bins=bins,
        bins_type=bins_type,
        metric_function=lambda y_true, y_pred: [
            len(y_true),
            sum([e != f for e, f in zip(y_true, y_pred)]),
        ],
    )
    misclass_rate_histogram = [metric_list[i][1] for i in range(len(metric_list))]

    if categorical is False:

        total_histogram_counts, bins = np.histogram(test_dataset[feature], bins)

    else:
        # Histogram of all points for categorical features
        total_histogram_counts = test_dataset[feature].value_counts().sort_index()

        total_histogram_counts = np.asarray(total_histogram_counts)

    rate = []
    standard_deviation = []
    for index in range(len(total_histogram_counts)):
        if total_histogram_counts[index] != 0:
            index_rate = misclass_rate_histogram[index] / total_histogram_counts[index]
            rate += [index_rate]
            if misclass_rate_histogram[index] != 0:
                standard_deviation += [
                    index_rate
                    * (
                        total_histogram_counts[index] ** (-1.0 / 2)
                        + misclass_rate_histogram[index] ** (-1.0 / 2)
                    )
                ]
            else:
                standard_deviation += [float("nan")]
        else:
            rate += [float("nan")]
            standard_deviation += [float("nan")]
    misclass_rate_histogram = rate

    return bins, misclass_rate_histogram, standard_deviation


def show_misclass_rate_feature(
    test_dataset,
    test_predictions,
    feature,
    categorical=False,
    bins=10,
    bins_type="regular",
    width_fraction=1.0,
    show_sd=False,
):
    """Displays the misclassification rate for the values of a certain feature.
    Parameters:
        test_dataset: Dataset with the features of all data points, where the
            true class is at the last column.
        test_predictions: List of the predicted classes for all data points.
        feature: Column name in the dataset of the feature.
        categorical (bool): Indicates whether it is a categorical feature.
            Default is "False".
        bins (int, list):
            * If an integer, this indicates the number of bins (default value is
            10). Whether this corresponds to dividing the feature scale in
            regularly spaced bins (default) or into quantiles, it must be
            specified in the parameter "bins_type".
            * If the string "quartiles" or "deciles" is used, then it
            automatically computes the appropriate bin edge postions to optimize
            for a quartile or decile grouping.
            * If any other feature intervals are needed, then a list of the
            feature values corresponding to the positions separating the bins
            and including the outermost edges must be provided.
        bins_type (str): If the bins parameter is an integer with the number
            of bins, this parameter allows to specify whether these bins should
            be "regular" evenly spaced bins or "quantiles". Default value is
            "regular".
        width_fraction (float): Fraction of the bin occupied by the bar.
        show_sd (bool): Whether the graph should display the standard deviation.
            Default is "False".
    """
    result_edges, result_rate, result_sd = misclass_rate_feature(
        test_dataset, test_predictions, feature, categorical=categorical, bins=bins
    )
    if categorical is False:
        width = np.diff(result_edges)
        width_interval = [bin * width_fraction for bin in width]
        result_edges = result_edges[:-1]
        alignment = "edge"
    else:
        result_edges = [str(item) for item in result_edges]
        alignment = "center"
        width_interval = 1

    plt.ylim(0, 1)
    plt.xlabel(feature)
    plt.ylabel("Misclassification rate")
    if show_sd:
        plt.bar(
            result_edges,
            result_rate,
            yerr=result_sd,
            width=width_interval,
            bottom=None,
            align=alignment,
            edgecolor="white",
            linewidth=2,
        )
    else:
        plt.bar(
            result_edges,
            result_rate,
            width=width_interval,
            bottom=None,
            align=alignment,
            edgecolor="white",
            linewidth=2,
        )
    plt.show(block=False)


def compute_conditional_metric(
    test_dataset,
    test_predictions,
    feature,
    categorical=False,
    bins=10,
    bins_type="regular",
    metric_function=lambda y_true, y_pred: f1_score(
        y_true, y_pred, pos_label="True", zero_division=0
    ),
):

    """Computes the conditional metrics.
    This function allows to compute the conditional metrics of a sample for
    the values of any particular feature.
    The function allows any binning for this calculation, which means that
    regularly spaced binnings, disparately spaced binnings that correspond to
    sets of an equal amount of data points (such as quartiles, deciles, or
    n-quantiles), or any other arbitrary irregular binning can be used.
    When the full dataset with all points does not have any data point in an
    interval corresponding to a certain bin, the function yields a "0" value
    for the metric.
    Parameters:
        test_dataset: Dataset with the features of all data points, where the
            true class is at the last column.
        test_predictions: List of the predicted classes for all data points.
        feature: Column name in the dataset of the feature.
        categorical (bool): Indicates whether it is a categorical feature.
            Default is "False".
        bins (int, list, str):
            * If an integer, this indicates the number of bins (default value is
            10). Whether this corresponds to dividing the feature scale in
            regularly spaced bins (default) or into quantiles, it must be
            specified in the parameter "bins_type".
            * If the string "quartiles" or "deciles" is used, then it
            automatically computes the appropriate bin edge postions to optimize
            for a quartile or decile grouping.
            * If any other feature intervals are needed, then a list of the
            feature values corresponding to the positions separating the bins
            and including the outermost edges must be provided.
        bins_type (str): If the bins parameter is an integer with the number
            of bins, this parameter allows to specify whether these bins should
            be "regular" evenly spaced bins or "quantiles". Default value is
            "regular".
        metric_function(lambda function): This parameter will take in a lambda function
            which will work on matched true y value and predicted y value. Please make sure
            to write a runnable metric score function(from sklearn.metric) with all parameters
            (with its value) you want to set. This misclass_rate_feature function will work
            exactly the metric function which you give on binned data.
            Such as: metric_function=lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label="True", zero_division=0)

    Returns:
        Two elements which correspond to:
            1) The edges of the bins in the feature scale.
            2) The list of conditional metrics in each bin.
    """

    y_test_name = list(test_dataset.columns)[-1]

    y_pred_series = pd.Series(test_predictions, test_dataset.index)

    bins = feature_binning(
        test_dataset=test_dataset,
        feature=feature,
        categorical=categorical,
        bins=bins,
        bins_type=bins_type,
    )

    if categorical is False:
        groups = test_dataset.groupby(
            by=pd.cut(
                test_dataset[feature], bins=bins, include_lowest=True, duplicates="drop"
            )
        )
        # compute metric
        metric_list = []
        # get all groups and compute their metrics for each group
        for key in groups.groups.keys():
            try:
                theGroup = groups.get_group(key)
                test_y = pd.Series(theGroup[y_test_name]).to_numpy()
                pred_y = y_pred_series[theGroup.index]
                sc = metric_function(test_y, pred_y)
                metric_list.append(sc)
            except KeyError:
                sc = metric_function([0], [0])
                metric_list.append(sc)

    else:
        # compute metric
        groups = test_dataset.groupby(by=feature)

        metric_list = []
        # get all groups and compute their metrics for each group
        for key in groups.groups.keys():
            theGroup = groups.get_group(key)
            test_y = pd.Series(theGroup[y_test_name]).to_numpy()
            pred_y = y_pred_series[theGroup.index]
            sc = metric_function(test_y, pred_y)
            metric_list.append(sc)

    return bins, metric_list


def show_misclass_rates_features(
    test_dataset, test_predictions, bins=10, show_sd=False
):
    """Displays the misclassification rate for the values of each feature.
    Parameters:
        test_dataset: Dataset with the features of all data points, where the
            true class is at the last column.
        test_predictions: List of the predicted classes for all data points.
        bins (int, list):
            * If an integer, it divides the feature scale in regularly spaced
            bins (default value is 10).
            * If the string "quartile" is used, then it automatically computes
            the appropriate bin edge postions to optimize for a quartile
            grouping.
            * If any other feature intervals are needed, then a list of the
            feature values corresponding to the positions separating the bins
            and including the outermost edges must be provided.
        show_sd (bool): Whether the graph should display the standard deviation.
            Default is "False".
    """
    # List of features
    feature_list = list(test_dataset.columns)[:-1]

    # Computes position of bin edges for quartiles or deciles
    for feature in feature_list:
        show_misclass_rate_feature(
            test_dataset, test_predictions, feature, bins=bins, show_sd=show_sd
        )


def compute_quantiles(dataset, feature, quantiles=4):
    """Computes optimal feature values to obtain n-quantiles.
    This function tries to determine the optimal feature value ranges in order
    to obtain groups of data of similar sizes (i.e. with an equal amount of
    samples), despite corresponding to feature intervals of different sizes.
    Very often this is not strictly possible. In particular, when the precision
    of the feature is small and many data points share the same feature values
    (i.e. the feature behaves as pseudo-discrete). In this case, these large
    subsets of data points sharing the same value either get all counted in one
    bin or they get all counted in another. Which makes it impossible to
    perfectly equilibrate the different groups.
    To arbitrarily split between two contiguous bins a subset of data points
    with the same feature value is not acceptable if different histograms and
    distributions have to be compared, or if normalization or other operations
    among them have to be carried out.
    Parameters:
        dataset (DataFrame): Data to try to chop in equal sets.
        feature: Column name in the dataset of the feature.
        quantiles (int): Number of equally-sized groups into which to try to
            divide the sample. For quartiles use 4, for deciles use 10, etc.
            Default value is 4.
    Returns:
        edge_values (list): List of the optimal edge positions.
    """
    list_quantiles = np.arange(quantiles + 1) / quantiles

    edge_values = np.quantile(dataset[feature], list_quantiles)
    return edge_values


def show_quantiles_feature(dataset, feature, quantiles=4, width_fraction=1.0):
    """Plots the best attempt to obtain n-quantiles for a feature.
    This function shows the different quantiles computed for one of the features
    in order to assess whether the data that is being used really allows for
    that particular number of quantiles to have a similar size or not.
    Parameters:
        dataset (DataFrame): Data to try to chop in equal sets.
        feature: Column name in the dataset of the feature.
        quantiles (int): Number of equally-sized groups into which to try to
            divide the sample. For quartiles use 4, for deciles use 10, etc.
            Default value is 4.
        width_fraction (float): Fraction of the bin occupied by the bar.
    """
    quantiles_feature = compute_quantiles(dataset, feature, quantiles=quantiles)
    total_histogram = np.histogram(dataset[feature], bins=quantiles_feature)
    width = [
        total_histogram[1][index + 1] - total_histogram[1][index]
        for index in range(len(total_histogram[0]))
    ]
    width_interval = [bin * width_fraction for bin in width]

    plt.xlabel(feature)
    plt.ylabel("counts")
    plt.bar(
        total_histogram[1][:-1],
        total_histogram[0],
        width=width_interval,
        bottom=None,
        align="edge",
        edgecolor="white",
        linewidth=3,
    )
    plt.show(block=False)


def show_quantiles_features(test_dataset, quantiles=4, width_fraction=1.0):
    """Plots the best attempt to obtain n quantiles for all features.
    This function shows the different quantiles computed for each one of the
    features in order to assess whether the data that is being used really
    allows for that particular number of quantiles to have a similar size for
    that feature or not.
    Parameters:
        dataset (DataFrame): Data to try to chop in equal sets.
        quantiles (int): Number of equally-sized groups into which to try to
            divide the sample. For quartiles use 4, for deciles use 10, etc.
            Default value is 4.
    """
    # List of features
    feature_list = list(test_dataset.columns)[:-1]

    for feature in feature_list:
        show_quantiles_feature(
            test_dataset, feature, quantiles=quantiles, width_fraction=width_fraction
        )


def feature_binning(
    test_dataset, feature, categorical=False, bins=10, bins_type="regular"
):
    """This function does binning based on any required features.
    This function can do regularly spaced binnings, disparately spaced binnings
    that corresponds to sets of an equal amount of data points
    (such as quartiles, deciles, or n-quantiles),
    or any other arbitrary irregular binning can be used.

    Parameters:
        test_dataset: Dataset with the features of all data points, where the
            true class is at the last column.
        test_predictions: List of the predicted classes for all data points.
        feature: Column name in the dataset of the feature.
        categorical (bool): Indicates whether it is a categorical feature.
            Default is "False".
        bins (int, list, str):
            * If an integer, this indicates the number of bins (default value is
            10). Whether this corresponds to dividing the feature scale in
            regularly spaced bins (default) or into quantiles, it must be
            specified in the parameter "bins_type".
            * If the string "quartiles" or "deciles" is used, then it
            automatically computes the appropriate bin edge postions to optimize
            for a quartile or decile grouping.
            * If any other feature intervals are needed, then a list of the
            feature values corresponding to the positions separating the bins
            and including the outermost edges must be provided.
        bins_type (str): If the bins parameter is an integer with the number
            of bins, this parameter allows to specify whether these bins should
            be "regular" evenly spaced bins or "quantiles". Default value is
            "regular".

    Returns:
            The edges of the bins in the feature scale.
    """

    if categorical is False:

        # Computes position of bin edges and groups for quartiles or deciles
        if bins == "quartiles":
            bins = compute_quantiles(test_dataset, feature, quantiles=4)

        elif bins == "deciles":
            bins = compute_quantiles(test_dataset, feature, quantiles=10)

        elif type(bins) == int and bins_type == "quantiles":
            bins = compute_quantiles(test_dataset, feature, quantiles=bins)

        # groups for regular bin edges or known bin edges
        else:
            total_histogram_counts, bins = np.histogram(test_dataset[feature], bins)

    else:

        # Histogram of all points for categorical features
        total_histogram_counts = test_dataset[feature].value_counts().sort_index()

        bins = np.asarray(total_histogram_counts.index)

    return bins
