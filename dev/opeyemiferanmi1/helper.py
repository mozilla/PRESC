#import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

def data_description(data):
    """Displays some basic descriptions of the imported data"""
    print(pd.DataFrame(data.head().T))
    print("\n")
    print('Shape of the data: ' + str(data.shape) + "\n")
    print("\n")
    print('Data description',(pd.DataFrame(data.describe())))
    print("\n")
    print('Vehicle unique Classes', data.Class.unique(), "\n")

def scale_fit_transform_data(df, column_list):
    """Takes in a dataframe and a list of column names to transform
        returns a dataframe of scaled values"""
    df_to_scale = df[column_list]
    x = df_to_scale.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    df_to_scale = pd.DataFrame(x_scaled, columns=df_to_scale.columns)
    return df_to_scale

def spot_check_algorithm(x_train, y_train, n_splits=10 ):
    """Spot checks different algorithms showing accuracy
     mean and standard deviation"""
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVC', SVC(gamma='auto')))
    models.append(('RFC', RandomForestClassifier()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=n_splits, random_state=42)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

def split(x, y, split_ratio=0.2):
    """Split the train and test data"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=1)
    return x_train, x_test, y_train, y_test


def train_SVC(x_train, y_train, x_test,y_test):
    """Train model with SVC"""
    svc_model = SVC(gamma='auto')
    svc_model.fit(x_train, y_train)
    y_pred =pd.DataFrame(svc_model.predict(x_test))
    return y_pred


def print_results(y_test, y_pred):
    """Print Model Accuracy"""
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy:",accuracy * 100)
    print(classification_report(y_pred, y_test))
    print(confusion_matrix(y_pred, y_test, ))
