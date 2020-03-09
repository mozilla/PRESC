import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report , confusion_matrix
    

def dataset_statistics(data):
    
    """ Data Features and Labels"""
    
    print("DATASET STATISTICS: \n")
    print("Shape of the dataset: ",data.shape)

    print("\nFeatures of the dataset are: ",list(data.keys()))
    
    print("\nTarget Features of dataset are: ", data['Class'].unique())
    
    

def data_visualization(data , label):
    
    """ For Visulaization of DataSet"""

    print(data.head(5))
    print("\nHistogram for analyzing the frequency of labled class\n")
    sns.countplot(data['Class'])

    print("Correlation Analysis.")
    plt.figure(figsize = (25,15))
    sns.heatmap(data.corr() , annot = True , linewidths=.5)
    plt.show()

    
def splitting_train_test_data(data):
    
    """ Data is splitted into 30:70 for training and testing"""
    X = data.drop(['Class'] , axis = 1)
    y  = data['Class']
    return train_test_split( X, y, test_size=0.3 , random_state=45)




def SVM_train_data(X,y):
    
    """ SVM Classifier"""
    
    classifier = SVC(gamma = 1 , kernel = 'poly', degree = 2)
    classifier.fit(X,y)
    
    
def evaluating_model(classifier , X_test):
    
    """ Evaluating model by predicting on testing data """
    
    y_predict = classifier.predict(X_test)  
    return y_predict
 
    
def model_confusion_matrix(y_test , y_predict , label):
    
    print("Confusion Matrix")
    cmatrix = confusion_matrix(y_test , y_predict , labels = label)       
    plt.figure(figsize = (15,5))
    plt.xlabel('Actual Vehicle Labels')
    plt.ylabel('Predicted Vehicle Labels')
    sns.heatmap(cmatrix, index = label , columns = label , annot=True, linewidths=.5)
    plt.show()

def model_classification_report(y_test, y_predict):
    """  Model Classification report for Precision , Recall and F1-Score """
    print("\n DataSet Report: ")
    print(classification_report(y_test, y_predict))
    