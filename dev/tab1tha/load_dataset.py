import pandas as pd
from pathlib import Path


def load_dataset(filename):
    """Function that reads the dataset file into a dataframe irrespective of
    the operating system. It returns a pandas dataframe. """
    datasets = Path("../../datasets/")
    fil = datasets / filename
    df = pd.read_csv(fil)
    return df
