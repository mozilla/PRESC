""" This file compares various evaluation metrics for different data splits """

import numpy as np
from dataloader import train_test_split_data
from evaluation import evaluate
from classifiers import Classifier
import pandas as pd
from IPython.display import HTML
from pylab import *

""" for now KNeighbors will be used as it gave the highest accuracy """
model = Classifier()

test_sizes = np.arange(0.0001,1,0.05)
columns = ['Training data','Testing Data','Accuracy', 'Precision', 'Recall', 'F1_score']
df = pd.DataFrame(columns = columns)

def data_split_examine():
    for index in range(len(test_sizes)):
        X_train, X_test, y_train, y_test = train_test_split_data(test_sizes[index])
        classifier = model.KNeighbors(X_train, y_train)
        accuracy, precision, recall, f_score, y = evaluate(classifier, X_test, y_test)
        train = round((1-test_sizes[index])*100)
        test = round(test_sizes[index]*100)
        df.loc[index+1] = [train, test, accuracy, precision, recall, f_score]

    display(df)
        
def visualise_split():
    fig,axes = plt.subplots()
    axes.set_xlabel("Accuracy")
    axes.set_ylabel("Test Data Size")
    axes.set_title("Relation btw accuracy and test data size")
    disp = axes.plot(test_sizes, df.Accuracy)
