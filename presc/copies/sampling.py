import numpy as np
import pandas as pd
from presc.dataset import Dataset


def dynamical_range(df):
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
    """Sample the classifier with a grid-like sampling."""
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
    """Sample the classifier with a random uniform sampling."""
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
    """Sample the classifier with a normal distribution sampling (with independent features)."""
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

    df_labeled = X.copy()

    # Label synthetic data with original classifier
    df_labeled[label_col] = original_classifier.predict(df_labeled)

    # Instantiate dataset wrapper
    df_labeled = Dataset(df_labeled, label_col=label_col)

    return df_labeled
