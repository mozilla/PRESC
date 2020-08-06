import numpy as np
import matplotlib.pyplot as plt


def misclass_rate_feature(
    test_dataset, test_predictions, feature, bins=10, bins_type="regular"
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
    # Computes position of bin edges for quartiles or deciles
    if bins == "quartiles":
        bins = compute_quantiles(test_dataset, feature, quantiles=4)
    elif bins == "deciles":
        bins = compute_quantiles(test_dataset, feature, quantiles=10)
    elif type(bins) == int and bins_type == "quantiles":
        bins = compute_quantiles(test_dataset, feature, quantiles=bins)

    # Histogram of all points
    total_histogram_counts, bins = np.histogram(test_dataset[feature], bins)

    # Builds dataset with only the misclassified data points
    test_dataset_misclass = test_dataset[test_dataset.iloc[:, -1] != test_predictions]

    # Histogram of misclassified points
    misclass_histogram_counts, bins = np.histogram(test_dataset_misclass[feature], bins)

    # Compute misclassification rate

    # The standard deviation in a counting experiment is N^(1/2).
    # According to error propagation the error of a quotient X=M/N is:
    # ErrorX = X(ErrorM/M + ErrorN/N),
    # here, Error_rate = rate*(M^(-1/2)+N^(-1/2))

    misclass_rate_histogram = np.copy(misclass_histogram_counts)

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
    bins=10,
    bins_type="regular",
    width_fraction=1.0,
):
    """Displays the misclassification rate for the values of a certain feature. 

    Parameters:
        test_dataset: Dataset with the features of all data points, where the 
            true class is at the last column.
        test_predictions: List of the predicted classes for all data points.
        feature: Column name in the dataset of the feature.
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
    """
    result_edges, result_rate, result_sd = misclass_rate_feature(
        test_dataset, test_predictions, feature, bins=bins
    )
    width = np.diff(result_edges)
    width_interval = [bin * width_fraction for bin in width]
    plt.ylim(0, 1)
    plt.xlabel(feature)
    plt.ylabel("Misclassification rate")
    plt.bar(
        result_edges[:-1],
        result_rate,
        yerr=result_sd,
        width=width_interval,
        bottom=None,
        align="edge",
        edgecolor="white",
        linewidth=2,
    )
    plt.show(block=False)


def show_misclass_rates_features(test_dataset, test_predictions, bins=10):
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
    """
    # List of features
    feature_list = list(test_dataset.columns)[:-1]

    # Computes position of bin edges for quartiles or deciles
    for feature in feature_list:
        show_misclass_rate_feature(test_dataset, test_predictions, feature, bins=bins)


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
    factor = 1.0 / quantiles
    list_quantiles = [0.0]
    for tile in range(quantiles):
        list_quantiles += [factor * (1 + tile)]

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
