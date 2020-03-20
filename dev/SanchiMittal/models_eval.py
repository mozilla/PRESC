"""
Train and Test the ML model, followed by its perormance evaluation.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def LogReg(x_train, x_test, y_train):
    """
    Train the classifier model on training set using Logisitc Regression.
    Get the predictions using the classifier model on test set.
    """
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)  # predicted target values
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
    print(accuracy_score(y_test, y_pred), "%")
    print("\n\n")

    # Performance of classification model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
