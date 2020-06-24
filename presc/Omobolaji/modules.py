import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Different models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def statistics(data):
    """View Information on the Dataset"""
    print("STATISTICS \n")
    print("Shape of data: ", data.shape, " \n")
    print("Columns in Data: \n", data.columns, " \n")
    print("Categories of Target Column: \n", data.Class.unique(), " \n")
    print(
        "Number of each category in Target Column: \n", data.Class.value_counts(), " \n"
    )
    print("Total number of null values in each column: \n", data.isnull().sum(), " \n")
    print("Description of each continuous feature: \n", data.describe().T, " \n")

def visualization(data):
    """Graphical Information on the Dataset"""
    print("An histogram showing categories in Target column, Class \n")
    sns.countplot(data=data, x="Class")
    plt.show()

    print("Correlation Analysis of Vehicle dataset \n")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    sns.heatmap(data.corr(), annot=True)
    plt.show()

def preprocessing(data):
    """Convert categorical column to continuous"""
    le = LabelEncoder()
    data["Class"] = le.fit_transform(data["Class"])
    """Split the columns into features and target"""
    features = data.drop(["Class"], axis=1)
    target = data.Class
    return train_test_split(features, target, test_size=0.3, random_state=42)

def model(model, train_features, test_features, train_target, test_target):
    """Applying a model on the dataset"""

    # fit the training data into the model
    model.fit(train_features, train_target)

    # store the predicted values of test data
    predict_model = model.predict(test_features)

    # do a cross validation on the model
    CV = 5
    cv_results = cross_val_score(
        model, train_features, train_target, cv=CV, scoring="accuracy"
    )
    cv = round(cv_results.mean(), 4), round(cv_results.std(), 4)
    print("Cross Validation(mean, standard deviation): ", cv, " \n")

    # Evaluate accuracy score
    accuracy = accuracy_score(test_target, predict_model)
    print("Accuracy Score(%): ", accuracy * 100, " \n")

    # Compute confusion matrix
    print("Confusion Matrix \n", confusion_matrix(test_target, predict_model), " \n")

    # Calculate the classification report
    print("Classification Report \n", classification_report(test_target, predict_model))
