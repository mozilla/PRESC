import pandas as pd
import sklearn
import seaborn as sns
import k_nn
from sklearn.model_selection import train_test_split

# import data into a pandas dataframe
df = pd.read_csv("..\..\datasets\generated.csv")

# EDA exploratory analysis of the data
# Exploration
print("*" * 50 + "HEAD" + "*" * 50)
print(df.head())
print("*" * 50 + "DESCRIBE" + "*" * 50)
print(df.describe())
print("*" * 50 + "INFO" + "*" * 50)
print(df.info())
print("*" * 100)

# todo visualisation

data = df.drop("label", axis=1)
target = df["label"]
# split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.30, random_state=10, stratify=target
)

# Evaluation of the  performance of the K-nearest neighbors prediction model
kn_accuracy = k_nn.k_nearest(data_train, target_train, data_test, target_test)
print(
    "The accuracy of the k nearest neighbours algorithm on this dataset is {}".format(
        kn_accuracy
    )
)
print(
    "The behaviour of the predicted values with respect to the actual values is as shown below:"
)
k_nn.visual_compare(data_train, target_train, data_test, target_test)
