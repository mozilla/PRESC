import numpy as np
import pandas as pd
from presc.dataset import Dataset
from presc.evaluations.utils import is_discrete


def dynamical_range(df):
    """Returns the dynamic range, mean, and sigma of the dataset features.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset with all the numerical features to analyze.

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


def find_categories(df, add_NaNs=False):
    """Returns the categories of the dataset features.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset with all the categorical features to analyze.
    add_NaNs : bool
        If True the sampler adds a "NaNs" category for the features that have
        any null values and assigns it the appropriate fraction.

    Returns
    -------
    dict of dicts
        A dictionary with an entry per dataset feature (dictionary keys are the
        column names), where each feature entry contains a nested dictionary
        with its categories and the fraction of each category present in the
        analyzed dataset (the nested dictionary key for this information is
        "categories", which is also a dictionary with one entry per category).
    """
    categories_dict = {}
    for feature in df:
        if is_discrete(df[feature]):
            # Remove NaN values from selection
            df_no_nans = df[df[feature].notnull()]

            # Log fraction of NaN values if required
            if add_NaNs:
                nan_fraction = df[feature].isnull().sum() / len(df)
                total_length = len(df)
            else:
                nan_fraction = 0
                total_length = len(df_no_nans)

            categories_dict[feature] = {
                "categories": {
                    key: None for key in df_no_nans[feature].unique().tolist()
                }
            }
            for category in categories_dict[feature]["categories"].keys():
                categories_dict[feature]["categories"][category] = (
                    df_no_nans[feature].value_counts()[category] / total_length
                )
            if add_NaNs and nan_fraction != 0:
                categories_dict[feature]["categories"]["NaNs"] = nan_fraction

    return categories_dict


def build_equal_category_dict(feature_categories):
    categories_dict = {}
    for feature, categories in feature_categories.items():
        categories_dict[feature] = {
            "categories": {key: 1 / len(categories) for key in categories}
        }
    return categories_dict


def mixed_data_features(df, add_NaNs=False):
    features_dict = {}
    for feature in df:
        df_feature = df[[feature]]
        if is_discrete(df[feature]):
            single_feature_parameters = find_categories(df_feature, add_NaNs=add_NaNs)
        else:
            single_feature_parameters = dynamical_range(df_feature)
        features_dict[feature] = single_feature_parameters[feature]

    return features_dict


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


def spherical_balancer_sampling(
    nsamplesxclass=1000,
    nfeatures=30,
    original_classifier=None,
    max_iter=10,
    nbatch=10000,
    radius_min=0,
    radius_max=1,
    label_col="class",
    random_state=None,
    verbose=False,
):
    """Sample the classifier with a balancer spherical distribution sampling.

    Generates synthetic samples with a spherical (shell) distribution between a
    minimum and a maximum radius values and then labels them using the original
    classifier. This function will attempt to obtain a balanced dataset by
    generating the same number of samples for all classes (`nsamplesxclass`),
    unless it reaches the maximum number of iterations. When used within the
    ClassifierCopy class, the `balancing_sampler` must be set to True.

    Parameters
    ----------
    nsamplesxclass : int
        Number of samples to generate per class.
    nfeatures : int
        Number of features of the generated samples.
    original_classifier : sklearn-type classifier
        Original ML classifier used to generate the synthetic data.
    max_iter : int
        The maximum number of iterations generating batches to attempt to obtain
        the samples per class specified in `nsamplesxclass`.
    nbatch: int
        Number of tentative samples to generate in each batch.
    radius_min : float
        Minimum radius of the spherical shell distribution. It will be a
        spherical distribution if this value is set to zero.
    radius_max : float
        Maximum radius of the spherical (shell) distribution.
    label_col : str
        Name of the label column.
    random_state : int
        Random seed used to generate the sampling data.
    verbose : bool
        If True the sampler prints information about each batch.

    Returns
    -------
    pandas DataFrame
        Dataset with a generated sampling following a spherical distribution of
        the feature space, with features and labels.
    """
    if random_state is not None:
        np.random.seed(seed=random_state)

    if verbose:
        print(
            f"\nGenerating samples between => min: {radius_min} and max: {radius_max}"
        )

    df_generated = pd.DataFrame()

    for iteration in range(max_iter):
        if verbose:
            print("Generating batch", iteration + 1)

        # Generate `nbatch` normalised vectors in random directions
        v = np.random.multivariate_normal(
            np.zeros((nfeatures,)), np.eye(nfeatures, nfeatures), size=nbatch
        )
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Scale the vectors between `radius_min` and `radius_max`
        alpha = np.random.uniform(low=radius_min, high=radius_max, size=nbatch)
        df_batch = pd.DataFrame(
            np.dot(alpha[:, np.newaxis], np.ones((1, nfeatures))) * v
        )
        # Label synthetic data with original classifier
        df_batch[label_col] = original_classifier.predict(df_batch)

        # Temporarily, add samples from the new batch to the old dataframe
        df_temp = pd.concat([df_generated, df_batch])

        # Keep a maximum of `nclass_samples` samples from each class
        detected_classes = df_temp[label_col].value_counts()
        df_generated = pd.DataFrame()
        for class_name in detected_classes.index.tolist():
            df_generated = pd.concat(
                [
                    df_generated,
                    df_temp[df_temp[label_col] == class_name].iloc[:nsamplesxclass],
                ]
            )

        # If there are no incomplete classes finish iteration, otherwise show classes
        incomplete_classes = detected_classes[
            detected_classes < nsamplesxclass
        ].sort_index()
        if len(incomplete_classes) == 0:
            return df_generated
        elif verbose:
            print("\nClasses:", incomplete_classes.index.tolist())
            print("Samples:", incomplete_classes.tolist())

    return df_generated


def categorical_sampling(nsamples=500, random_state=None, feature_parameters=None):
    """."""
    if random_state is not None:
        np.random.seed(seed=random_state)

    # Generate random data with the probabilities specified on the "categories"
    # entry of the feature_parameters
    X_generated = pd.DataFrame()
    for feature in feature_parameters:
        categories = list(feature_parameters[feature]["categories"].keys())
        category_probabilities = [
            feature_parameters[feature]["categories"][category]
            for category in categories
        ]
        X_generated[feature] = pd.Series(
            np.random.choice(categories, p=category_probabilities, size=nsamples)
        ).astype("category")

    return X_generated


def mixed_data_sampling(
    numerical_sampling, nsamples=500, random_state=None, feature_parameters=None
):
    """."""
    if random_state is not None:
        np.random.seed(seed=random_state)

    # Generate the lists of numerical and categorical data
    features_numerical = []
    features_categorical = []
    for feature in feature_parameters:
        if "categories" in feature_parameters[feature]:
            features_categorical.append(feature)
        else:
            features_numerical.append(feature)

    feature_parameters_numerical = {
        feature: feature_parameters[feature] for feature in features_numerical
    }
    feature_parameters_categorical = {
        feature: feature_parameters[feature] for feature in features_categorical
    }

    X_generated_numerical = numerical_sampling(
        nsamples=nsamples,
        random_state=random_state,
        feature_parameters=feature_parameters_numerical,
    )
    X_generated_categorical = categorical_sampling(
        nsamples=nsamples,
        random_state=random_state,
        feature_parameters=feature_parameters_categorical,
    )
    X_generated = pd.DataFrame()
    for feature in feature_parameters:
        if "categories" in feature_parameters[feature]:
            X_generated = X_generated.join(
                X_generated_categorical[[feature]], how="outer"
            )
        else:
            X_generated = X_generated.join(
                X_generated_numerical[[feature]], how="outer"
            )

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
    df_labeled[label_col] = df_labeled[label_col].astype("category")

    # Instantiate dataset wrapper
    df_labeled = Dataset(df_labeled, label_col=label_col)

    return df_labeled
