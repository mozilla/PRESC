import sklearn
import seaborn as sns
import k_nn
from load_dataset import load_dataset
from sklearn.model_selection import train_test_split
import explore_data

# import data into a pandas dataframe
filename = "generated.csv"
df = load_dataset(filename)

"""data preprocessing involves changing column names to more relatable ones in order to ease the interpretation of pair plots"""
df.columns = ["feature0", "feature1", "feature2", "feature3", "label"]
# change value names of the target column

# EDA exploratory analysis of the data
explore_data.raw(df)

# visualisation using pair plot and violin plot
explore_data.graph_visualize(df)
explore_data.violin_visualize(df)

data = df.drop("label", axis=1)
target = df["label"]
# split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.30, random_state=10, stratify=target
)
"""The standard default value of test_size is 0.25. However, 0.30 was chosen arbitrarily as a close approximation. 
The value of test_size has an effect on the performance of the model. This behaviour is observed in detail in pull request #43"""

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
