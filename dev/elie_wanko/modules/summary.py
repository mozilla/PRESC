import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import ravel, mean
from pandas import DataFrame
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

import helpers, knn, logreg
import timeit

def ttsplits(
    headers=[
        "Test Size",
        "Accuracy Score",
        "Precision Score",
        "Recall Score",
        "F1 Score",
        "Log Loss",
        "Av. Processing Time(sec.)"
    ],
    *,
    data,
    sizes,
    model
):
    """Builds a table of distribution of scores at different train:test splits
    
    data: Data set under consideration (target variable should be the last column
    sizes: set of train:test sizes to iterate over 
    model: model used to compute result
    
    returns: Dataframe (table) of results per the header argument.
    """

    # Split independent and target features
    independ_attrs = data.filter(data.columns[:-1])
    target_attrs = ravel(data.filter(data.columns[-1:]))  # Returns a 1-D array, containing the elements of the input

    # Evaluation metrics
    eval_metrics = []
    for s in sizes:
        # Split train and test data subsets
        subsets = train_test_split(independ_attrs, target_attrs, test_size=s, random_state=1)
        
        # Classifiers
        if model == "knn":
            # Processing starts
            start = timeit.default_timer()
            values_pred, values_true = knn.classifier(subsets=subsets, n_neighbors=7, threshold=0.35)
        elif model == "logreg":
            # Processing starts
            start = timeit.default_timer()
            values_pred, values_true = logreg.classifier(tol=0.000014, subsets=subsets, data=data, solver="liblinear", random_state=1)
        else:
            return "**Incorrect model chosen.**\n\nEnter * 'knn' for  K Nearest Neighbors(KNN)\n* 'logreg' for Logistic Regression\n* Otherwise, chose one from sklearn and enter in the following format 'model = sklearn.[...]'"

        
        # "Accuracy Score"
        ac = float("{:.4f}".format(metrics.accuracy_score(values_true, values_pred)))
        # "Precision Score"
        ps = float("{:.4f}".format(metrics.precision_score(values_true, values_pred)))
        # "Recall Score"
        rs = float("{:.4f}".format(metrics.recall_score(values_true, values_pred)))
        # "F1 Score"
        f1 = float("{:.4f}".format(metrics.f1_score(values_true, values_pred)))
        # "log_loss"
        ll = float("{:.4f}".format(metrics.log_loss(values_true, values_pred)))
        
        # Processing ends
        end = timeit.default_timer()

        # Evaluated metrics
        eval_metrics.append([s, ac, ps, rs, f1, ll, (end-start)])

    return DataFrame(eval_metrics, range(len(sizes)), headers)


def cv_folds(
    headers=[
        "K Folds",
        "Accuracy Score",
        "Precision Score",
        "Recall Score",
        "F1 Score",
        "Log Loss",
        "Av. Processing Time(sec.)"
    ],
    *,
    data,
    sizes,
    model
):
    """Builds a table of distribution of scores at different kfolds splits
    
    data: Data set under consideration (target variable should be the last column
    sizes: set of kfolds to iterate over
    model: model used to compute result
    
    returns: Dataframe (table) of results per the header argument.
    """

    # Split train and test data subsets
    independ_attrs = data.filter(data.columns[:-1])
    target_attrs = ravel(data.filter(data.columns[-1:]))  # Return a 1-D array, containing the elements of the input
    
    # Classifier and Scores
    if model =='knn':
        model = KNeighborsClassifier(n_neighbors = 7)
    elif model == 'logreg':
        model = LogisticRegression(tol=0.000014, solver="liblinear")
    else:
        return "**Incorrect model chosen.**\n\nEnter * 'knn' for  K Nearest Neighbors(KNN)\n* 'logreg' for Logistic Regression\n* Otherwise, chose one from sklearn and enter in the following format 'model = sklearn.[...]'"

    # Evaluated metrics
    eval_metrics = []
    for s in sizes:
        # Processing starts
        start = timeit.default_timer()
        
        # "Accuracy Score"
        ac = mean(cross_val_score(model, independ_attrs, target_attrs, cv=s, scoring="accuracy"))
        # "Precision Score"
        ps = mean(cross_val_score(model, independ_attrs, target_attrs, cv=s, scoring="precision"))
        # "Recall Score"
        rs = mean(cross_val_score(model, independ_attrs, target_attrs, cv=s, scoring="recall"))
        # "F1 Score"
        f1 = mean(cross_val_score(model, independ_attrs, target_attrs, cv=s, scoring="f1"))
        # "Log Loss"
        ll = mean(cross_val_score(model, independ_attrs, target_attrs, cv=s, scoring="neg_log_loss"))
        
        # Processing ends
        end = timeit.default_timer()
        
        
        # Arrary of results ( We divide the processing time (start-end) by 5 here because our cross_val_score is calculated 4 times with different scorings i.e. "Accuracy Score", "Precision Score", "Recall Score" and "F1 Score".
        eval_metrics.append([s, ac, ps, rs, f1, -ll, (end-start)/5])

    return DataFrame(eval_metrics, range(len(sizes)), headers)



    