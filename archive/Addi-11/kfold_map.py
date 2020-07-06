""" This file maps the evaluation metrics for various splits in the K-fold space"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from classifiers import Classifier
from evaluation import evaluate
from dataloader import get_X_y
from IPython.display import HTML

""" for now KNeighbors will be used as it gave the highest accuracy """
model = Classifier()

columns = ['Accuracy %','Precision %','Recall','F1_Score']
df = pd.DataFrame(columns = ['K_Fold']+columns)

""" Cross Validation using K-Fold """
def KFold_validation(kf, split_no):
    X, y = get_X_y()
    # kf.get_n_splits(X)
    for train, test in kf.split(X,y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y[train], y[test]

        classifier = model.KNeighbors(X_train, y_train)
        accuracy, precision, recall,f_score, y_score = evaluate(classifier, X_test, y_test)
    
    return  accuracy*100, precision*100, recall,f_score

""" Evaluation metric for different K-folds are shown in tabular format """
def tabulate_kfold():
    index = 1
    for split_no in range(2,80):
        kf = KFold(n_splits=split_no)
        df.loc[index,'K_Fold'] = split_no
        df.loc[index,columns] = KFold_validation(kf, split_no)
        
        index += 1

    display(df)