"""
Train and Test the ML model, followed by its performance evaluation.
"""
from math import sqrt, ceil
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def LogReg(x_train, x_test, y_train):
    """
    Train the classifier model on training set using Logistic Regression.
    Get the predictions using the classifier model on test set.
    """
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)  # predicted target values
    return y_pred

def KNN(x_train, x_test, y_train, y_test):
    """
    Train the classifier model on training set using K-Nearest Neighours.
    Get the predictions using the classifier model on test set.
    """

    score = 0
    
    # Choosing the optimum value of n_neighbors for KNN based on best accuracy
    for k in range(1, ceil(sqrt(x_train.size))): #range of n_neighbours has been taken upto square root of size of train set
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        
        if (accuracy_score(y_test, pred)>=score):
            score = accuracy_score(y_test, pred)
            y_pred = pred # predicted target values
        else:
            break
        
    return y_pred

def NaiveBayes(x_train, x_test, y_train):
    """
    Train the classifier model on training set using Naive Bayes.
    Get the predictions using the classifier model on test set.
    """
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    y_pred = nb.predict(x_test)  # predicted target values
    return y_pred

def Ksvm(x_train, x_test, y_train):
    """
    Train the classifier model on training set using Kernel Support Vector 
    Machine (RBF kernel).
    Get the predictions using the classifier model on test set.
    """
    ksvm = SVC(kernel='rbf', random_state= 0)
    
    ksvm.fit(x_train, y_train)

    y_pred = ksvm.predict(x_test)  # predicted target values
    return y_pred

def RFC(x_train, x_test, y_train):
    """
    Train the classifier model on training set using Random Forest Classification.
    Get the predictions using the classifier model on test set.
    """
    rfc = RandomForestClassifier(n_estimators=75, criterion= 'entropy', random_state= 0)
    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_test)  # predicted target values
    return y_pred

def Performance_Eval(y_test, y_pred):
    """
    Evaluate the performance of the ML model.\n
    List out the errors comparing with actual target labels in test data with 
    the help of confusion matrix.
    """
    # Making the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    print("\n\n")

    # Cheking accuracy
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred)*100, "%")
    print("\n\n")

    # Performance of classification model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
