import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score

    

def dataset_statistics(data):
    
    """ Data Features and Labels"""
    
    print("Shape of the dataset: ",data.shape)

    print("\nFeatures of the dataset are: ",list(data.keys()))
    
    print("\nTarget Labels of dataset are: ", data['Class'].unique())
    
    

def data_visualization(dataa):
    
    """ For Visulaization of DataSet"""
    
    print("\nHistogram for analyzing the frequency of labled class.\n")
    base_color = sns.color_palette()[9]
    Vorder = dataa['Class'].value_counts().index
    sns.countplot(data = dataa, x = 'Class', color = base_color , order = Vorder)
    
    print("\n")
    plt.figure(figsize = (25,15))
    sns.heatmap(dataa.corr() ,annot = True , linewidths=.5)
    print("Correlation Analysis.")
    plt.show()

    
def splitting_train_test_data(data):
    
    """ Data is splitted into 30:70 for training and testing"""
    #X = data.drop(['Class'] , axis = 1)
    #y  = data['Class']
    X = data.drop(['Class','Class_code'], axis = 1)
    y = data[['Class_code']]
    return train_test_split( X, y, test_size=0.3 , random_state=45)




def SVM_train(X,y):
    
    """ SVM Classifier"""
    # kernel =' poly ' is taking infinte time that's why it is not added.
    parameters = [{'kernel': ['rbf'],
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]
                    }]
    
    svm_grid = GridSearchCV(SVC(), parameters, cv=5)
    svm_grid.fit(X,y)
    classifier = SVC(kernel = svm_grid.best_estimator_.kernel ,
                     gamma = svm_grid.best_estimator_.gamma ,
                     C = svm_grid.best_estimator_.C )   
    
    return classifier.fit(X,y)



def LogisticRegression_train(X,y):
    
    """ SVM Classifier"""
    classifier = LogisticRegression()
    return classifier.fit(X,y)

    
def Tuning_LogiticRegression(X,y):
    """ Tuning Logistic Regression to increase the accuracy of model """    
    
    parameters = {'penalty': ['l1', 'l2'],
                  'C':[0.001,.009,0.01,.09,1,5,10,25]
                  }
    LR_grid = GridSearchCV(LogisticRegression(), param_grid = parameters , cv = 5)
    LR_grid = LR_grid.fit(X, y)
    classifier = LogisticRegression(penalty = LR_grid.best_estimator_.get_params()['penalty'],
                                    C = LR_grid.best_estimator_.get_params()['C']
                                    )
    return classifier.fit(X, y)
    
    
    
def test_classifier(classifier , X_test):
    
    """ Evaluating model by predicting on testing data """
    
    y_predict = classifier.predict(X_test)  
    return y_predict
 
def cross_validation(X,y,classifier):
    scores = cross_val_score( classifier, X, y, cv=5, scoring='f1_macro')
    return scores.mean()
    

def model_confusion_matrix(y_test , y_predict , dataa):    
    
    """ Drawing Confusion Matrix """
    
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    
    target_label = dataa['Class_code'].unique()
    target = dataa['Class'].unique()

    matrix = confusion_matrix(y_test, y_predict, labels = target_label)
    cmatrix = pd.DataFrame(matrix, index=target, columns=target)
    sns.heatmap(cmatrix, annot=True , linewidths=.5)
    
    plt.title("Confusion Matrix for Logistic Regression \n")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def model_classification_report(y_test, y_predict):

    """  Model Classification report for Precision , Recall and F1-Score """
    
    print("\nDataSet Report: ")
    print(classification_report(y_test, y_predict))
    
    


def randomize_data(X, Y):
    """ Randomize the labels and features data for learning curve"""
    
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation,:]
    Y2 = Y[permutation]
    return X2, Y2

def draw_learning_curves(X, y, estimator, num_trainings):
    
    """ Method to draw learning curves of different models """
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.grid()

    plt.title(" Learning Curves ")
    plt.xlabel(" Training Score ")
    plt.ylabel(" Score ")

    plt.plot(train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(test_scores_mean, 'o-', color="b", label="Cross-validation score")


    plt.legend(loc="best")

    plt.show()
    
    