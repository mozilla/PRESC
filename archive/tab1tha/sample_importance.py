"""This program evaluates the effect of a training datapoint on the performance of a KNN model.
It evaluates the performance of the model with and without the training data point of interest.
The data point of interest is chosen by the user. A more interactive, user-friendly version of the 
program can be made. Also, adjustments can be made such that the effect of two or more datapoints 
can be evaluated at the same time."""

# import necessary modules
import k_nn
from load_dataset import load_dataset
import explore_data

# import data into a pandas dataframe
filename = "vehicles.csv"
df = load_dataset(filename)

"""The observation whoose relevance needs to be evaluated is indicated by assigning its index to 
index_num. The 842nd observation was used here. The user can change it as is deemed fit. """
index_num = 842
df_without = df.drop(df.index[index_num])

# observe the initial dataset
print(df.tail())
# observe dataset when observation has been deleted
print(df_without.tail())

"""The accuracy of the performance model which was trained with and that which was trained without 
the observation is returned as a list. The elements of the list are in that order"""
lst = k_nn.with_without(df, df_without, "Class")

print(
    "The accuracy of the KNN model before the index {} observation was removed was {}.".format(
        index_num, lst[0]
    )
)
print(
    "The accuracy of the model without observation {} is {}.".format(index_num, lst[1])
)

difference = lst[0] - lst[1]
# The impact of the observation is calculated as a percentage and rounded up to two decimal places.
percentage = (difference / lst[0]) * 100
percentage = round(percentage, 2)

print("-" * 100)
print(
    "The performance score changed by {} when observation {} was not considered.".format(
        difference, index_num
    )
)
print(
    "The observation at index {} accounts for {} percent of the model's performance.".format(
        index_num, percentage
    )
)
