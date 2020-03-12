from sklearn.svm import SVC # SVM Classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV 

def train_svm(X, y, ratio = 0.3):
    """ This function takes train data, labels as input and trains the SVM model.

    Args:
        X: Training Data(features columns).
        y: Labels for each row
        ratio: (optional) Split ratio for train and test data

    Returns:
        tuple: y_test, y_pred values
    """
    
    ## Split the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = 1)
    
    ## Build the SVM model on training data
    model_svc = SVC(gamma='auto')
    model_svc.fit(X_train,y_train)
    y_pred= model_svc.predict(X_test)
    
    
    return y_test, y_pred
    
def train_svm_with_hyperparameter_tuning(X, y, param_grid, ratio = 0.3):
    """ This function takes train data, labels as input and trains the SVM model.

    The function uses GridSearchCV for hyperparameter tuning for SVM
    
    Args:
        X: Training Data(features columns).
        y: Labels for each row
        param_grid (json): params for GridSearchCV (c, gamma, kernel)
        ratio: (optional) Split ratio for train and test data

    Returns:
        tuple: y_test, y_pred values

    """
    
    ## Split the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = 1)

    svm_grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
    svm_grid.fit(X_train, y_train) 
    y_pred = svm_grid.predict(X_test) 
    
    return y_test, y_pred
    
def print_results(y_test, y_pred):
    """ This function takes y_test and y_pred values and print the results
    
    Args:
        y_test: Training Data(features columns).
        y_pred: Labels for each row

    Returns:
        None

    """
    
    # Print Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))