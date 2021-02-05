import pandas as pd
from sklearn.model_selection import train_test_split


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
        self._dataset = df
        self._X = None
        self._y = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        self.set_label(label)

    def set_label(self, label: str) -> None:
        """This function allows users to reset the label column and will also update the feature columns.

        Make sure you re-split the test and train data if you called this function after calling split.

        Args:
            label (string): name of the label column
        """
        try:
            self._X = self._dataset.drop(columns=[label])
            self._y = self._dataset[label]
        except KeyError:
            print(
                "Please make sure that the label you speficy is in the columns:"
                + ", ".join(self._dataset.columns.tolist())
            )
            raise

    def split_test_train(self, test_size: float = 0.2, random_state: int = 0) -> None:
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, test_size=test_size, random_state=random_state
        )

    @property
    def raw_dataset(self) -> pd.DataFrame:
        """Returns the underlying dataset."""
        return self._dataset

    @property
    def features(self) -> pd.DataFrame:
        """Returns the full dataset feature columns."""
        return self._X

    @property
    def feature_names(self) -> list:
        """Returns the feature names as a list."""
        return list(self.features.columns)

    @property
    def labels(self) -> pd.Series:
        """Returns the full dataset label column."""
        return self._y

    @property
    def train_features(self) -> pd.DataFrame:
        """Returns the feature columns for the training set portion."""
        return self._X_train

    @property
    def train_labels(self) -> pd.Series:
        """Returns the label column for the training set portion."""
        return self._y_train

    @property
    def test_features(self) -> pd.DataFrame:
        """Returns the feature columns for the test set portion."""
        return self._X_test

    @property
    def test_labels(self) -> pd.Series:
        """Returns the label column for the test set portion."""
        return self._y_test

    def get_test_dataset(self) -> pd.DataFrame:
        dataset_test = self._X_test.merge(
            self._y_test, left_index=True, right_index=True, how="left"
        )
        return dataset_test

    def get_train_dataset(self) -> pd.DataFrame:
        dataset_train = self._X_train.merge(
            self._y_train, left_index=True, right_index=True, how="left"
        )
        return dataset_train
