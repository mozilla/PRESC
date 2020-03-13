# importing necessary libraries
import os
import pandas as pd
import numpy as np
import itertools
import pandas_profiling
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,\
classification_report,roc_curve,confusion_matrix

class DefineModules():
    '''
    Defines the different functions that will be called during
    execution
    '''

    # reading the data
    def read_data(folder, file_name):
        '''
        This function reads the csv file from the folder it is stored in 
        and returns a dataframe.

        Parameters: main directory, file name

        Returns: dataframe    
        '''
        data = pd.read_csv('../../{}/{}.csv'.format(folder, file_name))

        return data

    # studying the data
    def study_data(data):
        '''
        This function helps know the various details of the data like, 
        number of null values, skewness, distribution, correlations, 
        missing values, interactions, data type, unique values etc.

        Parameters: dataframe

        Returns: report object


        '''
        profile = pandas_profiling.ProfileReport(data)

        return profile
    
    # pre-processing
    def preprocessing(data):
        '''
        This function drops duplicates present in the dataframe and
        converts the target column from object --> int datatype.
        
        Parameters: dataframe
        
        Returns: dataframe
        '''
        
        # dropping duplicates
        data.drop_duplicates(inplace=True)
        
        # converting data['recommend'] to integer column
        data.recommend = data.recommend*1
        
        return data
    
    # splitting the data into train-test
    def data_split(data, t_size, rndm_state):
        '''
        This function performs train-test split on the dataframe
        depending on the test-size and random-state specified.
        
        Parameters: dataframe, test-size (int) and random-state (int)
        
        Returns: four dataframes namely- X_train, y_train, X_test,
        y_test
        '''

        # defining features (x) columns
        data_x = data.drop(columns = ['recommend'], axis=1)

        #defining target (y) column
        data_y = data['recommend']

        # performing train-test split
        X_train, X_test, y_train, y_test = train_test_split(
                                        data_x,
                                        data_y,
                                        test_size = t_size,
                                        random_state = rndm_state
        )

        return X_train, X_test, y_train, y_test
    
    # applying stochastic gradient descent
    def sgd_model(X_train, y_train):
        '''

        This function fits SGD model on the input dataframes
        and calculates the top 15 features that influence the model
        performance
        
        Parameters: two dataframes (X_train, y_train)
        
        Returns: model
        '''
        
        # creating SGD instance
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10)
        
        # fitting the model on train data
        clf.fit(X_train, y_train)

        return clf
    
    # evaluation metric
    def scoring(y_test, X_test, clf):
        '''
        This function predicts on X_test and returns the
        classification report and confusion matrix
        
        Parameters: two dataframes (y_test, X_test)and fitted model
        
        Returns: object explaining classification report
        '''
        
        # store the predicted values of test data
        y_pred = clf.predict(X_test)
        
        # calculating classification report
        eval_report = classification_report(y_test,y_pred)
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        return eval_report, cnf_matrix
    
    #Evaluation of Model - Confusion Matrix Plot
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        
        Parameters: confusion_matrix (object),
        classes (here, classes=['Recommend (True)', 'Not Recommend (False)']),
        normalize (boolean)
        """
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()