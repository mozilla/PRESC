""" This file compares various evaluation metrics for different data splits """

import numpy as np
from dataloader import train_test_split_data
from evaluation import evaluate
from classifiers import Classifier
import pandas as pd
from IPython.display import HTML
from pylab import *

test_sizes = np.arange(0.005,1,0.05)
columns = ['Training data','Testing Data','Accuracy %', 'Precision', 'Recall', 'F1_score']
df = pd.DataFrame(columns = columns)

def data_split_examine(clf):
    model = Classifier()
    for index in range(len(test_sizes)):
        X_train, X_test, y_train, y_test = train_test_split_data(test_sizes[index])
        classifier = getattr(model, clf)(X_train, y_train) 
        accuracy, precision, recall, f_score, _ = evaluate(classifier, X_test, y_test)
        train = round((1-test_sizes[index])*100)
        test = round(test_sizes[index]*100)
        df.loc[index+1] = [train, test, accuracy*100, precision, recall, f_score]

    display(df)
        
def visualise_split():
    fig,axes = plt.subplots()
    axes.set_xlabel("Test Data Size")
    axes.set_ylabel("Accuracy %")
    axes.set_ylim([50,100])
    axes.set_title("Relation btw accuracy and test data size")
    disp = axes.plot(df['Testing Data'], df['Accuracy %'])
