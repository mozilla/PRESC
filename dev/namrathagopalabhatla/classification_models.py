import numpy as  np
import pandas as pd

import matplotlib.pyplot as plt 

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.linear_model import LogisticRegression

from seaborn import heatmap
from IPython.display import display

class MLP:
    def __init__(self, hidden_size=None, activation='relu', solver='lbfgs', max_iter=1000):
        '''Constructor for initalization of model using provided parameters'''
        super(MLP, self).__init__()
        if hidden_size is None:
            print("Hidden layer(s)' size(s) not specified.")
            return None
        self.classifier = MLPClassifier(hidden_size, solver=solver, activation=activation, max_iter=max_iter)
        self.predictions = None
        self.report = None
        self.confusion = None
    
    def classify(self, X):
        '''Function that returns prediction based on trained model'''
        if X is None:
            print('Data not provided.')
            return None

        prediction = self.classifier.predict(X)
        return prediction
        
    def train(self, X, y):
        '''Function to train the MLP model'''
        if X is None or y is None:
            print('Data not provided.')
            return False

        self.classifier.fit(X, y)
        return True
        
    def confusion_matrix(self, X=None, y=None):
        '''Function to show and return confusion matrix'''
        if self.confusion is None and X is None and y is None:
            print('Data not provided.')
            return None
        elif self.confusion is None and X is not None and y is not None:
            _, _ = self.validate(X, y)
        
        ax = plt.axes()
        heatmap(self.confusion, annot=True, ax=ax)
        ax.set_title('Confusion Matrix')
        plt.show()
        
        return self.confusion
    
    def show_report(self, X=None, y=None):
        '''Function to show and return classification report including accuracy, F1 score, precision, recall, etc.'''
        if self.report is None and X is None and y is None:
            print('Data not provided.')
            return None
        elif self.report is None and X is not None and y is not None:
            _, _ = self.validate(X, y)
        
        display(self.report)
        return self.report
        
    def validate(self, X, y):
        '''Function to validate test/validation set and evaluate accuracy'''
        if X is None or y is None:
            print('Data not provided.')
            return None, None

        self.predictions = self.classify(X)
        
        self.report = classification_report(y, self.predictions, output_dict=True)
        self.report = pd.DataFrame(self.report).transpose()
        
        self.confusion = confusion_matrix(y, self.predictions)
        return self.report, self.confusion

class Logistic:
    def __init__(self, solver='lbfgs', multi_class='auto'):
        '''Constructor for initalization of model using provided parameters'''
        super(Logistic, self).__init__()
        self.classifier = LogisticRegression(solver=solver, multi_class=multi_class)
        self.predictions = None
        self.report = None
        self.confusion = None
    
    def classify(self, X):
        '''Function that returns prediction based on trained model'''
        if X is None:
            print('Data not provided.')
            return None

        self.predictions = self.classifier.predict(X)
        return self.predictions
        
    def train(self, X, y):
        '''Function to train the Logistic Regression model'''
        if X is None or y is None:
            print('Data not provided.')
            return False

        self.classifier.fit(X, y)
        return True
        
    def confusion_matrix(self, X=None, y=None):
        '''Function to show and return confusion matrix'''
        if self.confusion is None and X is None and y is None:
            print('Data not provided.')
            return None
        elif self.confusion is None and X is not None and y is not None:
            _, _ = self.validate(X, y)
        
        ax = plt.axes()
        heatmap(self.confusion, annot=True, ax=ax)
        ax.set_title('Confusion Matrix')
        plt.show()
        
        return self.confusion
    
    def show_report(self, X=None, y=None):
        '''Function to show and return classification report including accuracy, F1 score, precision, recall, etc.'''
        if self.report is None and X is None and y is None:
            print('Data not provided.')
            return None
        elif self.report is None and X is not None and y is not None:
            _, _ = self.validate(X, y)
        
        display(self.report)
        return self.report
        
    def validate(self, X, y):
        '''Function to validate test/validation set and evaluate accuracy'''
        if X is None or y is None:
            print('Data not provided.')
            return None, None
            
        self.predictions = self.classify(X)
        
        self.report = classification_report(y, self.predictions, output_dict=True)
        self.report = pd.DataFrame(self.report).transpose()
        
        self.confusion = confusion_matrix(y, self.predictions)
        return self.report, self.confusion