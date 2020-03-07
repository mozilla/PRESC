from dataloader import train_test_split_data
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train, X_test,y_train, y_test = train_test_split_data()

def svm_classifier():
    classifier = SVC(gamma='auto')
    classifier.fit(X_train, y_train)
    return classifier

def KNeighbors():
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def Logistic_Reg():
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    return classifier

def Decision_Tree():
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def Random_Forest():
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    return classifier