import numpy as np
import pandas as pd
from presc.dataset import Dataset


def grid_sampling(
    nsamples=500,
    feature_parameters={"x0": (-1, 1), "x1": (-1, 1)},
):
    """Sample the classifier with a grid-like sampling."""
    # Compute number of points per feature (assuming same number of points)
    nfeatures = len(feature_parameters)
    npoints = int(nsamples ** (1 / nfeatures))

    # Generate grid

    feature_list = []
    feature_names = []
    for key in feature_parameters:
        feature_list.append(
            np.linspace(feature_parameters[key][0], feature_parameters[key][1], npoints)
        )
        feature_names.append(key)

    X_generated = pd.DataFrame()
    for index, item in enumerate(np.meshgrid(*feature_list)):
        X_generated[index] = item.ravel()
    X_generated.columns = feature_names

    return X_generated


def uniform_sampling(nsamples=500, feature_parameters={"x0": (-1, 1), "x1": (-1, 1)}):
    """Sample the classifier with a random uniform sampling."""
    # Generate random uniform data
    X_generated = pd.DataFrame()
    for key in feature_parameters:
        X_generated[key] = np.random.uniform(
            feature_parameters[key][0], feature_parameters[key][1], size=nsamples
        )

    return X_generated


def normal_sampling(
    nsamples=500,
    feature_parameters={"x0": (0, 1), "x1": (0, 1)},
    label_col="y",
):
    """Sample the classifier with a normal distribution sampling (with independent features)."""
    # Compute number of features
    nfeatures = len(feature_parameters)

    # Rename columns
    feature_names = []
    mus = []
    sigmas = []
    for key in feature_parameters:
        feature_names.append(key)
        mus.append(feature_parameters[key][0])
        sigmas.append(feature_parameters[key][1])

    mus = np.array(mus)
    covariate_matrix = np.eye(nfeatures, nfeatures) * (np.array(sigmas)) ** 2

    # Generate normal distribution data
    X_generated = pd.DataFrame(
        np.random.multivariate_normal(mus, covariate_matrix, size=nsamples)
    )

    # Rename columns
    X_generated.columns = feature_names

    return X_generated


def labeling(X, original_classifier, label_col="y"):

    df_labeled = X.copy()

    # Label synthetic data with original classifier
    df_labeled[label_col] = original_classifier.predict(df_labeled)

    # Instantiate dataset wrapper
    df_labeled = Dataset(df_labeled, label_col=label_col)

    return df_labeled
