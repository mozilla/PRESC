import pandas as pd
from sklearn.model_selection import train_test_split

"""
Dataset Wrapper API

Provides some functionalities to access the datasets as a pandas DataFrame object.
You can access the raw dataset as well as the the feature and label columns.
You can also split the dataset into train and test datasets, and access them.

"""


class DatasetWrapper(object):
    _dataset = None
    X, y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None

    def __init__(self, dataset_file: str, label: str) -> None:
        """
        Description:
            To initialize a dataset object using the link found in the datasets folder.

        Args:
            dataset_file (string): path to the dataset file to use.
            label (string) : name of the label column
        """

        try:
            # read the dataset from file path
            self._dataset = pd.read_csv(dataset_file)

            # set self.X and self.y by label column
            self.X = self._dataset.drop(columns=[label])
            self.y = self._dataset[label]

        except (FileNotFoundError, pd.errors.ParserError, KeyError):
            print(
                "\nPlease check the file path and column label that your passed is valid.\n"
                + "It should be in the format dataset/xxx.csv if you are currently in the PRESC folder.\n"
                + "The label column is specified in the README.md.\n"
            )
            raise

    def set_label(self, label: str) -> None:
        """
        Description:
            This function allows users to reset the label column and will also update the feature columns.

        Args:
            label (string): name of the label column
        """
        self.X = self._dataset.drop(columns=[label])
        self.y = self._dataset[label]

    def split_test_train(self, test_size: float = 0.2, random_state: int = 0) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def get_raw_dataset(self) -> object:
        return self._dataset

    def get_label(self, subset: str = None) -> object:
        """
        Description:
            This function allows user to get the label column of a dataset.
            By default, it will return the label column of the unsplit/raw dataset.
            If "test"/"train" is passed in the subset parameter, it will return the corresponding label column.

        Args:
            subset (str, optional): . Defaults to None.

        Raises:
            ValueError: If a string other than "test" and "train" is passed in, a ValueError is raised.

        Returns:
            object: label column of the specified dataset by the subset parameter.
        """
        if subset == "test":
            return self.y_test
        elif subset == "train":
            return self.y_train
        elif subset:
            raise ValueError("Subset dataset can only be test or train.")

        return self.y

    def get_features(self, subset: str = None) -> object:
        """
        Description:
            This function allows user to get the feature column(s) of a dataset.
            It works in a similar fashion as get_label(). See get_label() for more details.
        """
        if subset == "test":
            return self.X_test
        elif subset == "train":
            return self.X_train
        elif subset:
            raise ValueError("Subset dataset can only be test or train.")

        return self.X

    def get_test_dataset(self) -> object:
        dataset_test = self.X_test.merge(
            self.y_test, left_index=True, right_index=True, how="left"
        )
        return dataset_test

    def get_train_dataset(self) -> object:
        dataset_train = self.X_train.merge(
            self.y_train, left_index=True, right_index=True, how="left"
        )
        return dataset_train
