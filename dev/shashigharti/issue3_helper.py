from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def train_svm_with_hyperparameter_tuning(X, y, param_grid, ratio=0.3):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=1)

    svm_grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    svm_grid.fit(X_train, y_train)
    y_pred = svm_grid.predict(X_test)

    return y_test, y_pred