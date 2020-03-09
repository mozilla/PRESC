import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report , confusion_matrix
    


vehicle = pd.read_csv('C:/Users/pcs/Documents/GitHub/PRESC/datasets/vehicles.csv')

def dataset_statistics(data):
    print("DATASET STATISTICS: \n")
    print("Shape of the dataset: ",data.shape)

    print("\nFeatures of the dataset are: ",list(data.keys()))
    
    print("\nTarget Features of dataset are: ", data['Class'].unique())
    
    

def data_visualization(data):
    
    print(data.head(5))
    print("\nHistogram for analyzing the frequency of labled class\n")
    sns.countplot(vehicle['Class'])

    print("Correlation Analysis.")
    figure = plt.gcf()
    figure.set_size_inches(18, 10)
    sns.heatmap(vehicle.corr(), annot = True , linewidths=.5)
    plt.show()

    
def splitting_train_test_data(data):
    X = data.drop(['Class'] , axis = 1)
    y  = vehicle['Class']
    return train_test_split( X, y, test_size=0.3 , random_state=45)


def SVM_train_data(X,y):
    classifier = SVC(gamma = 1 , kernel = 'poly', degree = 2)
    classifier.fit(X,y)
    
    
def evaluating_model(classifier , X_test):
    y_predict = classifier.predict(X_test)  
    return y_predict
 
def model_confusion_matrix(y_test , y_predict , label):
    cmatrix = confusion_matrix(y_test , y_predict , labels = label)       
    figure = plt.gcf()
    figure.set_size_inches(10, 5)
    sns.heatmap(cmatrix, annot=True, linewidths=.5)
    plt.show()

def model_classification_report(y_test, y_predict):
    print(classification_report(y_test, y_predict))
    
dataset_statistics(vehicle)    
data_visualization(vehicle)