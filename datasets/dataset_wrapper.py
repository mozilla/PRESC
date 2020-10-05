import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
Dataset Wrapper API 

Provides some functionalities to access and preprocess the datasets.
functionalities: 
1. reading and writing from files - DONE
2. access train/test of the data  - DONE 
3. accessing feature columns
4. label column accessing pre-split train/test sets
5. TBD

"""
# Helper functions:

class DatasetWrapper:
    name = ""
    dataset = None

    X,y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None
    
    transformer = None
    X_train_transformed ,X_test_transformed = None, None

    def __init__(self,test_dataset_name:str) -> None:
        """
        Description:
            To initialize a dataset object. 
            The client may only choose a dataset from the already existed datasets.

        Args:
            test_dataset_name (string): a string that is the name of one of the test/datasets
        """  
        # Datasets from the datasets/ folder     
        self._datasets = ("generated", "vehicles", "eeg", "winequality", "defaults")

        if test_dataset_name not in self._datasets:
            raise ValueError("The test dataset name you chose was not in the datasets/ folder.")
        else:
            # TODO: do I always drop quality column? Or is it dataset specific
            self.name = test_dataset_name
        
        # read the dataset from url
        self.dataset = pd.read_csv("https://github.com/mozilla/PRESC/blob/master/datasets/" + self.name + ".csv")

        # set X,y
        self.X,self.y = self.dataset.iloc[:,:-1], self.dataset.iloc[:,-1]

    def split_test_train(self,test_size = 0.2, random_state = 0):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
        test_size=test_size, random_state=random_state)

    def get_label(self):
        return self.y

    def get_features(self):
        return self.X

    # TODO: should implement dataset-specific processor
    def get_feature_processer(self):
        self.transformer = StandardScaler().fit(self.X_train)
        return self.transformer

    def transform_dataset(self):
        self.X_train_transformed  = self.transformer.transform(self.X_train)
        self.X_test_transformed  = self.transformer.transform(self.X_test)

    def get_test_dataset(self):
        dataset_test = self.X_test.merge(self.y_test, 
        left_index=True, right_index=True, how='left') 
        return dataset_test

    
    
