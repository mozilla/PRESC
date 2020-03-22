import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import  f1_score, recall_score, precision_score, classification_report

from plot_confusionmatrix import plot_confusion_matrix

def model(X_train,y_train,X_test,y_test):
    '''
     Input:
           y_test is the target variable in the testing set.
           y_train is the target variable in the training set.
           X_test is the variable in the testing set.
           X_train is the  variable in the training set.

    Outputs: 
        returns the performance and the result of the model
    '''
    results={}
    def evaluate(clf):
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        target_names = ['van', 'saab', 'bus', 'opel']
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

        print(clf.__class__.__name__, ":\n", classification_report(y_pred, y_test))
        plot_cm = plot_confusion_matrix(cm, target_names, title='Confusion matrix')
        ### Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        f1__score = f1_score(y_pred, y_test, average='macro')
        precision = precision_score(y_pred, y_test, average='macro')
        recallscore = recall_score(y_pred, y_test, average='macro')
        return f1__score, precision, recallscore
    
    clf = LogisticRegression()
    results["Logistic"]=evaluate(clf)

    clf = RandomForestClassifier()
    results["RandomForest"] = evaluate(clf)
    
    clf = AdaBoostClassifier()
    results["AdaBoost"] = evaluate(clf)
    
    clf = SVC()
    results["SVC RBF"] = evaluate(clf)
    
    clf = SVC(kernel="linear")
    results["SVC Linear"] = evaluate(clf)
    
    clf = DecisionTreeClassifier()
    results['Decision Tree']  = evaluate(clf)
    print("\n")
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=['f1 score', 'precision', 'recall_score'] 
    return results

