import numpy as np
import pandas as pd

from presc.dataset import Dataset


def multiclass_gaussians(
    nsamples=3000,
    nfeatures=30,
    nclasses=15,
    center_low=2,
    center_high=10,
    scale_low=1,
    scale_high=1,
):
    """Generates a multidimensional gaussian dataset with multiple classes.

    This function generates a multidimensional normal distribution centered at
    the origin with standard deviation one for class zero. And then adds an
    additional gaussian distribution per class, centered at a random distance
    between `center_low` and `center_high`, and with random standard deviation
    between `scale_low` and `scale_high`.

    Parameters
    ----------
    nsamples : int
        Maximum number of samples to generate. Actual number of samples depends
        on the number of classes, because the function yields a balanced
        dataset with the same number of samples per class.
    nfeatures : int
        Number of features of the generated samples.
    nclasses : int
        Number of classes in the generated dataset.
    center_low : float
        Minimum translation from the origin of the center of the gaussian
        distributions corresponding to additional classes.
    center_high : float
        Maximum translation from the origin of the center of the gaussian
        distributions corresponding to additional classes.
    scale_low : float
        Minimum value for the standard deviation of the gaussian distributions
        corresponding to additional classes.
    scale_high : float
        Maximum value for the standard deviation of the gaussian distributions
        corresponding to additional classes.

    Returns
    -------
    presc.dataset.Dataset
        Outputs a PRESC Dataset with the generated samples and their labels.
    """
    class_samples = int(nsamples / nclasses)

    # Create class zero drawing samples from a `nfeatures`-dimensional normal
    # distribution centered at the origin and with a standard deviation between
    # `scale_low` and `scale_high`.
    scale = np.random.uniform(low=scale_low, high=scale_high)
    t_pred = scale * np.random.normal(0, 1, (class_samples, nfeatures))
    df_pred = pd.DataFrame(t_pred)
    df_pred["class"] = 0

    # Create additional classes centered at `m` with standard deviation `scale`
    for i in range(1, nclasses):
        # Generate a normalized vector in a random direction
        v = np.random.normal(0, 1, nfeatures)
        v = v / np.linalg.norm(v)

        # Generate a random distance from the origin to define the center of each gaussian
        alpha = np.random.uniform(low=center_low, high=center_high)
        m = alpha * v

        # Generate a random scaling for each gaussian
        scale = np.random.uniform(low=scale_low, high=scale_high)

        # Generate normally distributed random samples for this classs
        t = m + scale * np.random.normal(0, 1, (class_samples, nfeatures))
        df = pd.DataFrame(t)
        df["class"] = i

        # Add class data to the dataset
        df_pred = pd.concat([df_pred, df], ignore_index=True)

    # Convert into PRESC Dataset
    df_presc = Dataset(df_pred, label_col="class")
    return df_presc
