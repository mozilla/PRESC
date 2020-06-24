import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings


def test_train_data(data_X, data_y):
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X, data_y, test_size = 0.2)
    print(data_X_train.shape,data_X_test.shape,data_y_train.shape)
    
    return data_X_train, data_X_test, data_y_train, data_y_test


def linear_classifier(data_X_train, data_X_test, data_y_train, data_y_test):
'''
Creating the classifier, 
the inputs are my training data and that is used for predicting y values for test data
In-built function for an accuracy_score is used.
Linear Kernel
'''
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(data_X_train, data_y_train)
    data_y_pred = svclassifier.predict(data_X_test)
    print("SVM - linear:")
    print(accuracy_score(data_y_test,data_y_pred))
    return data_y_pred

def rbf_classifier(data_X_train, data_X_test, data_y_train, data_y_test):
'''
Creating the classifier, 
the inputs are my training data and that is used for predicting y values for test data
In-built function for an accuracy_score is used.
RBF Kernel
'''
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(data_X_train, data_y_train)
    data_y_pred = svclassifier.predict(data_X_test)
    print("SVM - rbf:")
    print(accuracy_score(data_y_test,data_y_pred))
    return data_y_pred

def sigmoid_classifier(data_X_train, data_X_test, data_y_train, data_y_test):
    '''
Creating the classifier, 
the inputs are my training data and that is used for predicting y values for test data
In-built function for an accuracy_score is used.
Sigmoid Kernel
'''
    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(data_X_train, data_y_train)
    data_y_pred = svclassifier.predict(data_X_test)
    print("SVM - sigmoid:")
    print(accuracy_score(data_y_test,data_y_pred))
    return data_y_pred

def knn_classifier(data_X_train, data_X_test, data_y_train, data_y_test):
    '''
Creating a K nearest neighbours Classifier
the inputs are my training data and that is used for predicting y values for test data
In-built function for an accuracy_score is used.
'''
    classifier = KNeighborsClassifier(n_neighbors=8)  
    classifier.fit(data_X_train, data_y_train)  
    data_y_pred = classifier.predict(data_X_test)  
    print('Accuracy - KNN ')
    print(accuracy_score(data_y_test,data_y_pred))
    return data_y_pred