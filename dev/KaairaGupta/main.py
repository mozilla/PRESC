from data_loader import load_split_preprocessed_data
from Ml_functions import model
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters

#take normal transaction in  0.5 , 0.66 and 0.75  proportion of total data


#LogisticRegression
for i in range(1,4):
    print("the model classification for {} proportion".format(i))
    undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=load_split_preprocessed_data(i)
    clf=LogisticRegression()
    model(clf,undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test)
    print("________________________________________________________________________________________________________")


#SVC
for i in range(1,4):
    print("the model classification for {} proportion".format(i))
    undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=load_split_preprocessed_data(i)
    clf= SVC()# here we are just changing classifier
    model(clf,undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test)
    print("________________________________________________________________________________________________________")

#Random Forest Classifier
for i in range(1,4):
    print("the model classification for {} proportion".format(i))
    undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=load_split_preprocessed_data(i)
    clf= RandomForestClassifier(n_estimators=100)# here we are just changing classifier
    model(clf,undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test)
    print("________________________________________________________________________________________________________")