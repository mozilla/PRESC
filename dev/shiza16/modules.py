
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
    

def dataset_statistics(data):
    
    """ Data Features and Labels"""
    
    print("DATASET STATISTICS: \n")
    print("Shape of the dataset: ",data.shape)

    print("\nFeatures of the dataset are: ",list(data.keys()))
    
    print("\nTarget Labels of dataset are: ", data['Class'].unique())
    
    

def data_visualization(dataa):
    
    """ For Visulaization of DataSet"""
    
    print("\nHistogram for analyzing the frequency of labled class.\n")
    base_color = sns.color_palette()[9]
    Vorder = dataa['Class'].value_counts().index
    sns.countplot(data = dataa, x = 'Class', color = base_color , order = Vorder)
    
    plt.figure(figsize = (25,15))
    print("Correlation Analysis.")
    sns.heatmap(dataa.corr() ,annot = True , linewidths=.5)
    plt.show()

    
def splitting_train_test_data(data):
    
    """ Data is splitted into 30:70 for training and testing"""
    X = data.drop(['Class'] , axis = 1)
    y  = data['Class']
    return train_test_split( X, y, test_size=0.2 , random_state=45)




def SVM_train(X,y):
    
    """ SVM Classifier"""
    
    classifier = SVC(gamma = 1 , kernel = 'poly', degree = 2)
    return classifier.fit(X,y)


def LogisticRegression_train(X,y):
    
    """ SVM Classifier"""
    classifier = LogisticRegression()
    return classifier.fit(X,y)
    
    
    
    
def test_classifier(classifier , X_test):
    
    """ Evaluating model by predicting on testing data """
    
    y_predict = classifier.predict(X_test)  
    return y_predict
 
    
def model_confusion_matrix(y_test , y_predict , dataa):    
    
    """ Confusion Matrix """
    
    plt.figure(figsize = (8,8))
    target_label = dataa['Class'].unique()
    plt.xlabel('Real Vehicle Category')
    plt.ylabel('Predicted Vehicle Category')

    matrix = confusion_matrix(y_test, y_predict, labels = target_label)
    cmatrix = pd.DataFrame(matrix, index=target_label, columns=target_label)
    sns.heatmap(cmatrix, annot=True, fmt='d')
    
    
    plt.show()

def model_classification_report(y_test, y_predict):
    """  Model Classification report for Precision , Recall and F1-Score """
    
    print("\n DataSet Report: ")
    print(classification_report(y_test, y_predict))
    
    