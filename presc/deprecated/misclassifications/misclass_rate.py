import numpy as np
import pandas as pd


# --------------------
# This should be kept for now, as we may want to use the standard deviation
# logic in the future.
#
# However, the functionality has been replaced by
# presc.evaluations.conditional_metric.compute_conditional_metric
# --------------------
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

    from presc.evaluations.conditional_metric import compute_conditional_metric as ccm

    y_test_name = list(test_dataset.columns)[-1]
    y_pred_series = pd.Series(test_predictions, test_dataset.index)

    if bins == "quartiles":
        bins = 4
        bins_type = "quantile"
    elif bins == "deciles":
        bins = 10
        bins_type = "quantile"

    result = ccm(
        test_dataset[feature],
        test_dataset[y_test_name],
        y_pred_series,
        as_categorical=categorical,
        num_bins=bins,
        quantile=bins_type.startswith("quantile"),
        metric=lambda y_true, y_pred: [
            len(y_true),
            sum([e != f for e, f in zip(y_true, y_pred)]),
        ],
    )
    bins = result.bins.to_numpy()
    metric_list = list(result.vals)

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
