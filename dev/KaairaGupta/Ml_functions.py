
from data_loader import load_split_preprocessed_data
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



def KNeighbors():
    X_train, X_test, y_train, y_test = load_split_preprocessed_data()
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred_knn = clf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred_knn)
    precision=metrics.precision_score(y_test, y_pred_knn)
    recall=metrics.recall_score(y_test, y_pred_knn)

    return accuracy, precision, recall

def LogisticR():
    X_train, X_test, y_train, y_test = load_split_preprocessed_data()
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    y_pred_lr = clf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred_lr)
    precision=metrics.precision_score(y_test, y_pred_lr)
    recall=metrics.recall_score(y_test, y_pred_lr)

    return accuracy, precision, recall

def DecisionT():
    X_train, X_test, y_train, y_test = load_split_preprocessed_data()
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred_dt = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred_dt)
    precision = metrics.precision_score(y_test, y_pred_dt)
    recall = metrics.recall_score(y_test, y_pred_dt)

    return accuracy, precision, recall

def RandomF():
    X_train, X_test, y_train, y_test = load_split_preprocessed_data()
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, y_train)
    y_pred_rfc = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred_rfc)
    precision = metrics.precision_score(y_test, y_pred_rfc)
    recall = metrics.recall_score(y_test, y_pred_rfc)

    return accuracy, precision, recall

