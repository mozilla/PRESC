"""Generate fake data to use for testing.

Test datasets generated using this code are stored as fixtures in the repo. This
code is included here for reference only.
"""

import sys
from string import ascii_uppercase

from sklearn.datasets import make_classification
from pandas import DataFrame


def make_test_dataset(pkl_filepath):
    """Creates a synthetic classification dataset to use for testing.

    Dataset is a pandas DataFrame written to pickle at the given path.
    """
    # Create a synthetic classification dataset
    X, y = make_classification(
        n_samples=100,
        # 5 features, 1 will be pure noise
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        # Assign 10% of labels at random to add noise
        flip_y=0.1,
        shuffle=False,
        random_state=543,
    )

    df = DataFrame(X, columns=["a", "b", "c", "d", "e"])
    # Convert one of the informative columns to categorical (string):
    # Round values to integer and map integers to letters
    to_categ = df["c"].astype("int")
    to_categ_uniq = sorted(to_categ.unique())
    categ = to_categ.map(
        dict(zip(to_categ_uniq, list(ascii_uppercase[: len(to_categ_uniq)])))
    )
    df["c"] = categ
    # Append the label column
    df["label"] = y
    df.to_pickle(pkl_filepath)


if __name__ == "__main__":
    make_test_dataset(sys.argv[1])
