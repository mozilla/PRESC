from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def CrossValidationFolds_Traversal(estimator, vdataset):
    """
    Arguments:
    - estimator
    - dataset
    This function computes acccuracy with the
    Cross Validation Score with KFold from 2 to 5 for each split proportion of
    training and testing set.
    
    """
    X = vdataset.drop(["Class", "Class_code"], axis=1)
    y = vdataset["Class_code"]
    scores = []
    matrix = pd.DataFrame(
        columns=["KFold", "Accuracy"]
    )
    for i in range(1, 11): ##Kfold 1 to 10
        score = cross_val_score(estimator, X, y, cv=i, scoring="accuracy")
        scores.append(score.mean())
        matrix = matrix.append(
            {
                "KFold": i,
                "Accuracy": (score.mean() * 100),
            },
            ignore_index=True,
        )
    return matrix


def Visulaize_CrossValidationFolds_Traversal(split_matrix):
    """
    Argument:
        - Dataframe named split_matrix
    Line Plot is drawn for each KFold value with it's respective performance score.
    
    """
    ax = plt.gca()
    print("------------------------------------------------------------------")
    split_matrix.plot(kind="line", x="KFold", y="Accuracy", color="red", ax=ax)
    split_matrix.plot(kind="line", x="KFold", y="Testing_Set", color="yellow", ax=ax)
    split_matrix.plot(kind="line", x="KFold", y="Training_Set", color="blue", ax=ax)
    plt.title("Line plot with  size = \n")
    plt.ylabel("Accuracy\n")
    plt.xlabel("\nNo of KFolds")
    plt.show()
