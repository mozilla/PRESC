import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report

# def load_data(filename):
#     ''' Loading dataset in pandas dataframe and printing first five data point'''
#     dataset = pd.read_csv(filename)
#     print(dataset.head())
    
def data_stats(dataset):
    ''' Shows some basic stats of the dataset'''
    print("=========== SOME STATS of Dataset ===========")
    print('Shape of the dataset: ' + str(dataset.shape))
    print('List of attribute columns' , list(dataset.columns))
    list_cat = dataset.Class.unique()
    print('List of Categories ' , list_cat )
    
def tokenize_target_column(dataset):
    ''' tokenize the Class column values to numeric data'''
    factor = pd.factorize(dataset['Class'])
    dataset.Class = factor[0]
    definitions = factor[1]
    print("Updated tokenize 'Class' column - first 5 values")
    print(dataset.Class.head())
    print("Distinct Tokens used for converting Class column to integers")
    print(definitions)
    return definitions
def train_data_test_data_split(dataset):
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:,-1].values
#     print(X[0])
#     print(y[0])
#     print(X.shape)
#     print(y.shape)
#     print('the data attributes columns')
#     print(X[:5,:])
#     print('The target variable: ')
#     print(y[:5])
#     Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 21)
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    ''' training model on train data'''
    classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)
    return classifier
    
def test(classifier, X_test):
    ''' testing model on test data'''
    y_pred=classifier.predict(X_test)
    return y_pred
    
def untokenizing_testdata_prediction(y_test, y_pred, definitions):
    '''Converting numeric target and predict values back to original labels'''
    reversefactor = dict(zip(range(4),definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    return y_test, y_pred


def create_confusion_matrix_class_report(y_test, y_pred):
    ''' Creates Cinfusion Matrix and summary of evaluation metric '''
    
    labels = ["van" , "saab" ,"bus" , "opel"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.xlabel('Real Vehicle Category')
    plt.ylabel('Predicted Vehicle Category')
    print("============== Summary of all evaluation metics ===============")
    print(classification_report(y_test,y_pred))
    print ("====================== Confusion Matrix=====================")
    
    