import pandas as pd
from sklearn.model_selection import train_test_split
from presc.dataset import Dataset


# --------------------
# This is no longer needed.
# It has been replaced by presc.dataset.Dataset
# Keeping here so as not to break the tests.
# --------------------
class TrainTestDataset:
    """Dataset API

    Provides some functionalities to access the datasets as a pandas DataFrame object.
    You can access the raw dataset as well as the the feature and label columns.
    You can also split the dataset into train and test datasets, and access them.

    Args:
        df (DataFrame): the underlying dataset
        label (string) : name of the label column
    """

    def __init__(self, df: pd.DataFrame, label: str) -> None:
        self._dataset = Dataset(df, label)
        self._train = None
        self._test = None

    def split_test_train(self, test_size: float = 0.2, random_state: int = 0) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self._dataset.features,
            self._dataset.labels,
            test_size=test_size,
            random_state=random_state,
        )
        df_train = pd.DataFrame(X_train, columns=self._dataset.feature_names)
        df_train[self._dataset._label_col] = y_train
        self._train = Dataset(df_train, self._dataset._label_col)

        df_test = pd.DataFrame(X_test, columns=self._dataset.feature_names)
        df_test[self._dataset._label_col] = y_test
        self._test = Dataset(df_test, self._dataset._label_col)

    @property
    def raw_dataset(self) -> pd.DataFrame:
        """Returns the underlying dataset."""
        return self._dataset.df

    @property
    def features(self) -> pd.DataFrame:
        """Returns the full dataset feature columns."""
        return self._dataset.features

    @property
    def feature_names(self) -> list:
        """Returns the feature names as a list."""
        return self._dataset.feature_names

    @property
    def labels(self) -> pd.Series:
        """Returns the full dataset label column."""
        return self._dataset.labels

    @property
    def train_features(self) -> pd.DataFrame:
        """Returns the feature columns for the training set portion."""
        return self._train.features

    @property
    def train_labels(self) -> pd.Series:
        """Returns the label column for the training set portion."""
        return self._train.labels

    @property
    def test_features(self) -> pd.DataFrame:
        """Returns the feature columns for the test set portion."""
        return self._test.features

    @property
    def test_labels(self) -> pd.Series:
        """Returns the label column for the test set portion."""
        return self._test.labels

    def get_test_dataset(self) -> pd.DataFrame:
        return self._test.df

    def get_train_dataset(self) -> pd.DataFrame:
        return self._train.df
