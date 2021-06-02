import numpy as np
import pandas as pd
from itertools import product
from presc.dataset import Dataset


def grid_sampling(
    classifier,
    nsamples=500,
    feature_parameters={"x0": (-1, 1), "x1": (-1, 1)},
    label_col="y",
):
    """Sample the classifier with a grid-like sampling."""
    # Compute number of points per feature (assuming same number of points)
    nfeatures = len(feature_parameters)
    npoints = int(nsamples ** (1 / nfeatures))

    # Generate grid

    X_points = pd.DataFrame()
    feature_list = []
    feature_names = []
    for key in feature_parameters:
        X_points[key] = np.linspace(
            feature_parameters[key][0], feature_parameters[key][1], npoints
        )
        feature_list += [X_points[key]]
        feature_names += [key]

    X_generated = pd.DataFrame.from_records(
        list(i for i in product(*feature_list)), columns=feature_names
    )

    # Label synthetic data with original classifier
    X_generated[label_col] = classifier.predict(X_generated)

    # Instantiate dataset wrapper
    X_generated = Dataset(X_generated, label_col=label_col)

    return X_generated


def uniform_sampling(
    classifier,
    nsamples=500,
    feature_parameters={"x0": (-1, 1), "x1": (-1, 1)},
    label_col="y",
):
    """Sample the classifier with a random uniform sampling."""
    # Generate random uniform data
    X_generated = pd.DataFrame()
    for key in feature_parameters:
        X_generated[key] = np.random.uniform(
            feature_parameters[key][0], feature_parameters[key][1], size=nsamples
        )

    # Label synthetic data with original classifier
    X_generated[label_col] = classifier.predict(X_generated)

    # Instantiate dataset wrapper
    X_generated = Dataset(X_generated, label_col=label_col)

    return X_generated


def normal_sampling(
    classifier,
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
        feature_names += [key]
        mus += [feature_parameters[key][0]]
        sigmas += [feature_parameters[key][1]]

    mus = np.array(mus)
    covariate_matrix = np.eye(nfeatures, nfeatures) * (np.array(sigmas)) ** 2

    # Generate normal distribution data
    X_generated = pd.DataFrame(
        np.random.multivariate_normal(mus, covariate_matrix, size=nsamples)
    )

    # Rename columns
    X_generated.columns = feature_names

    # Label synthetic data with original classifier
    X_generated[label_col] = classifier.predict(X_generated)

    # Instantiate dataset wrapper
    X_generated = Dataset(X_generated, label_col=label_col)

    return X_generated
