from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def Calculate_accuracy(classifier, X_train, y_train, size , shape ):
    """
    This function computes acccuracy with the
    Cross Validation Score with KFold from 2 to 5 for each split proportion of
    training and testing set.
    
    """
    scores = []
    split_matrix = pd.DataFrame(
        columns=["Training_Set", "Testing_Set", "Accuracy"]
    )
    
    loocv = LeaveOneOut()
    test_size = 100 * size
    train_size = 100 - (test_size)
    
    score = cross_val_score(classifier, X_train, y_train, cv=loocv, scoring="accuracy")
    scores.append(score.mean())
    split_matrix = split_matrix.append(
    {
         "Training_Set": train_size,
         "Testing_Set": test_size,
         "Accuracy": (score.mean() * 100),
    },
    ignore_index=True)

    return split_matrix



def visulaize_train_test_split_traversal(split_matrix):
    """
     Line Plot is drawn for each split proportion
     of traing and testing set with the performance score.
    
    """
    print("------------------------------------------------------------------")
   
    split_matrix.plot(x ='Training_Set', y='Accuracy', kind = 'line')
    
    plt.ylabel("Accuracy\n")
    plt.xlabel("\nTesting Set")
    plt.show()


def train_test_split_traversal(classifier, vdataset):

    """
    Arguments:
    - estimator
    - dataset
    
    this function computes an evaluation metric over a grid of train/test split 
    proportions from 20 to 80%, for each split proportion it resplit and
    recompute the metric multiple times.
    
    Output:
    DataFrame of splits with multiple metric values per split.
    
    """

    X = vdataset.drop(["Class", "Class_code"], axis=1)
    y = vdataset["Class_code"]
    shape =  X.shape[0]
    split_matrix = pd.DataFrame(
        columns=["Training_Set", "Testing_Set", "Accuracy"]
    )
    
    size = [0.1,0.15 ,0.2, 0.25 , 0.3, 0.35 ,0.4, 0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    for s in size:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=s, random_state=45
        )
        matrix = Calculate_accuracy(classifier, X_train, y_train, s , shape)
        split_matrix = pd.concat([split_matrix, matrix], ignore_index=True)

    return split_matrix
