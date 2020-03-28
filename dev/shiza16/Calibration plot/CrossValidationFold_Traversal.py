from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt


def CrossValidationFolds_Traversal(estimator, vdataset):
    """
    Arguments:    
        - estimator  = classifer of model
        - vdataset =  vehicld dataset 
    
    This function computes acccuracy score with the
    Cross Validation Score for each KFold with K from 2 to 10 
    
    Output:
        returns matrix conatining value of K with it's corresponding performance score
    """
    X = vdataset.drop(["Class", "Class_code"], axis=1)
    y = vdataset["Class_code"]
    scores = []
    matrix = pd.DataFrame(columns=["KFold", "Accuracy"])
    for i in range(2, 11):  ##Kfold 2 to 10
        score = cross_val_score(estimator, X, y, cv=i, scoring="accuracy")
        scores.append(score.mean())
        matrix = matrix.append(
            {"KFold": i, "Accuracy": (score.mean() * 100),}, ignore_index=True,
        )
    return matrix


def Visulaize_CrossValidationFolds_Traversal(matrix):
    """
    Argument:
        - matrix: Dataframe named matrix
    
    Line Plot is drawn for each KFold value with it's respective performance score.
    
    Output:
        - plot the line graph
    """
    ax = plt.gca()
    matrix.plot(
        kind="line",
        x="KFold",
        y="Accuracy",
        color="red",
        marker="o",
        markerfacecolor="blue",
        markersize=12,
        ax=ax,
    )
    plt.title("Line plot of No of Kfold with it's corresponding performance score\n")
    plt.ylabel("Accuracy\n")
    plt.xlabel("\nNo of KFolds")
    plt.show()
