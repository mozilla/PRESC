import pandas as pd
from sklearn.model_selection import train_test_split
"""
Dataset Wrapper API 

Provides some functionalities to access the datasets as a pandas DataFrame object.
You can access the raw dataset as well as the the feature and label columns.
You can also split the dataset into train and test datasets, and access them.

"""
class DatasetWrapper(object):
    X,y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None

    def __init__(self,dataset_file:str) -> None:
        """
        Description:
            To initialize a dataset object using the link found in the datasets folder 

        Args:
            test_dataset_name (string): a string that is the name of one of the test/datasets
        """
        # read the dataset from file path
        try:
            self._dataset = pd.read_csv(dataset_file)
        except IOError as e:
            print(e + ''' Please check the file path that your passed is correct. \n 
            It should be in the format dataset/xxx.csv if you are currently in the PRESC folder.''')

        # set X,y
        self.X,self.y = self._dataset.iloc[:,:-1], self._dataset.iloc[:,-1]

    def split_test_train(self,test_size:float = 0.2, random_state:int = 0) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
        test_size=test_size, random_state=random_state)

    def get_raw_dataset(self) -> object:
        return self._dataset

    def get_label(self) -> object:
        return self.y

    def get_features(self) -> object:
        return self.X

    def get_test_dataset(self) -> object:
        dataset_test = self.X_test.merge(self.y_test, 
        left_index=True, right_index=True, how='left') 
        return dataset_test

    def get_train_dataset(self) -> object:
        dataset_train = self.X_train.merge(self.y_train, 
        left_index=True, right_index=True, how='left') 
        return dataset_train