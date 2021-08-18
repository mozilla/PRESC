import numpy as np
import pandas as pd
from presc.dataset import Dataset


def dynamical_range(df):
    """Returns the dynamic range, mean, and sigma of the dataset features.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset with all the features to analyze.

    Returns
    -------
    dict of dicts
        A dictionary with an entry per dataset feature (dictionary keys are the
        column names), where each feature entry contains a nested dictionary
        with the values of the minimum and maximum values of the dynamic range
        of the dataset, as well as the mean and sigma of the distribution
        (nested dictionary keys are "min", "max", "mean" and "sigma").
    """
    range_dict = {}
    for feature in df:
        range_dict[feature] = {
            "min": df[feature].min(),
            "max": df[feature].max(),
            "mean": df[feature].mean(),
            "sigma": df[feature].std(),
        }

        print(
            f"{feature} min: {range_dict[feature]['min']:.4f}    "
            f"{feature} max: {range_dict[feature]['max']:.4f}    "
            f"{feature} mean: {range_dict[feature]['mean']:.4f}    "
            f"{feature} sigma: {range_dict[feature]['sigma']:.4f}    "
            f"Interval: {range_dict[feature]['max']-range_dict[feature]['min']:.4f}   "
        )

    return range_dict


def grid_sampling(nsamples=500, random_state=None, feature_parameters=None):
    """Sample the classifier with a grid-like sampling.

    Generates synthetic samples with a regular grid-like distribution within the
    feature space described in `feature_parameters`. Computes the grid spacing
    so that all features have the same number of different values.

    Parameters
    ----------
    nsamples : int
        Maximum number of samples to generate. The exact number will depend on
        the parameter space.
    random_state : int
        Parameter not used in `grid_sampling`.
    feature_parameters : dict of dicts
        A dictionary with an entry per dataset feature (dictionary keys should
        be the feature names), and where each feature entry must contain a
        nested dictionary with at least the entries corresponding to the minimum
        and maximum values of the dynamic range. Dictionary keys for these
        values should be "min" and "max", respectively.

    Returns
    -------
    pandas DataFrame
        Dataset with a regular grid-like generated sampling of the feature space
        characterized by the `feature_parameters`."""
    # Compute number of points per feature (assuming same number of points)
    nfeatures = len(feature_parameters)
    npoints = int(nsamples ** (1 / nfeatures))

    # Generate grid

    feature_list = []
    feature_names = []
    for key in feature_parameters:
        feature_list.append(
            np.linspace(
                feature_parameters[key]["min"], feature_parameters[key]["max"], npoints
            )
        )
        feature_names.append(key)

    X_generated = pd.DataFrame()
    for index, item in enumerate(np.meshgrid(*feature_list)):
        X_generated[index] = item.ravel()
    X_generated.columns = feature_names

    return X_generated


def uniform_sampling(nsamples=500, random_state=None, feature_parameters=None):
    """Sample the classifier with a random uniform sampling.

    Generates synthetic samples with a random uniform distribution within the
    feature space described in `feature_parameters`.

    Parameters
    ----------
    nsamples : int
        Number of samples to generate.
    random_state : int
        Random seed used to generate the sampling data.
    feature_parameters : dict of dicts
        A dictionary with an entry per dataset feature (dictionary keys should
        be the feature names), and where each feature entry must contain a
        nested dictionary with at least the entries corresponding to the minimum
        and maximum values of the dynamic range. Dictionary keys for these
        values should be "min" and "max", respectively.

    Returns
    -------
    pandas DataFrame
        Dataset with a random uniform generated sampling of the feature space
        characterized by the `feature_parameters`.
    """
    if random_state is not None:
        np.random.seed(seed=random_state)

    # Generate random uniform data
    X_generated = pd.DataFrame()
    for key in feature_parameters:
        X_generated[key] = np.random.uniform(
            feature_parameters[key]["min"],
            feature_parameters[key]["max"],
            size=nsamples,
        )

    return X_generated


def normal_sampling(
    nsamples=500,
    random_state=None,
    feature_parameters=None,
):
    """Sample the classifier with a normal distribution sampling.

    Generates synthetic samples with a normal distribution according to the
    feature space described by `feature_parameters`. Features are assumed to be
    independent (not correlated).

    Parameters
    ----------
    nsamples : int
        Number of samples to generate.
    random_state : int
        Random seed used to generate the sampling data.
    feature_parameters : dict of dicts
        A dictionary with an entry per dataset feature (dictionary keys should
        be the feature names), and where each feature entry must contain a
        nested dictionary with at least the entries corresponding to the mean
        and standard deviation values of the dataset. Dictionary keys for these
        values should be "mean" and "sigma", respectively.

    Returns
    -------
    pandas DataFrame
        Dataset with a generated sampling following a normal distribution of
        the feature space characterized by the `feature_parameters`.
    """
    if random_state is not None:
        np.random.seed(seed=random_state)

    # Compute number of features
    nfeatures = len(feature_parameters)

    # Rename columns
    feature_names = []
    mus = []
    sigmas = []
    for key in feature_parameters:
        feature_names.append(key)
        mus.append(feature_parameters[key]["mean"])
        sigmas.append(feature_parameters[key]["sigma"])

    mus = np.array(mus)
    covariate_matrix = np.eye(nfeatures, nfeatures) * (np.array(sigmas)) ** 2

    # Generate normal distribution data
    X_generated = pd.DataFrame(
        np.random.multivariate_normal(mus, covariate_matrix, size=nsamples)
    )

    # Rename columns
    X_generated.columns = feature_names

    return X_generated


def labeling(X, original_classifier, label_col="class"):
    """Labels the samples from a dataset according to a classifier.

    Parameters
    ----------
    X : pandas DataFrame
        Dataset with the features but not the labels.
    original_classifier : sklearn-type classifier
        Classifier to use for the labeling of the samples.
    label_col : str
        Name of the label column.

    Returns
    -------
    presc.dataset.Dataset
        Outputs a PRESC Dataset with the samples and their labels.
    """
    df_labeled = X.copy()

    # Label synthetic data with original classifier
    df_labeled[label_col] = original_classifier.predict(df_labeled)

    # Instantiate dataset wrapper
    df_labeled = Dataset(df_labeled, label_col=label_col)

    return df_labeled
