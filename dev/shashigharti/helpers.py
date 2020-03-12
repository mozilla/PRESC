from sklearn.svm import SVC # SVM Classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV 

def train_svm(X, y, ratio = 0.3):
    ## Split the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = 1)
    
    ## Build the SVM model on training data
    model_svc = SVC(gamma='auto')
    model_svc.fit(X_train,y_train)
    y_pred= model_svc.predict(X_test)
    
    return y_test, y_pred
    
def train_svm_with_hyperparameter_tuning(X, y, param_grid, ratio = 0.3):
    ## Split the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = 1)

    svm_grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
    svm_grid.fit(X_train, y_train) 
    y_pred = svm_grid.predict(X_test) 
    
    return y_test, y_pred
    
def print_results(y_test, y_pred):
    # Print Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))