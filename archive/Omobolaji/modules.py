import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

#Import preprocessing module & evaluation metrics
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


    
def test_size_analysis(dataset, models):
    """ This function takes a dataset and a dictionary of different models
    as params and fits the data into each model with different test sizes. 
    Returns result of the test sizes and accuracy score of all the models 
    in a tabular form.
    
    Args:
        dataset: Dataset
        models: Dictionary of different models
    Returns:
        results: A table of results with different test 
                sizes and the accuracy score for different models. 
    """
    
    #create an empty list
    results = list()
    
    #Split the dataset into features and target variables
    features = dataset.drop(["Class"], axis=1)
    target = dataset.Class
   
    # loop through the ranges of different test sizes from 0.1 to 0.5
    for split in range(10, 55, 5):
        split = split / 100  # convert to decimal format
        train_size = 1 - split
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=split, random_state=42
        )
        
        #create a list containing the test_sizes, that is, split
        result = [split]
        
        #loop through key and value, contained in models dictionary
        for key, model in models.items():
        
            # fit the training data into the model
            model.fit(train_features, train_target)

            # store the predicted values of test data
            predict_target = model.predict(test_features)

            # Evaluate accuracy score
            accuracy = accuracy_score(test_target, predict_target)

            # convert the score to a percentage and round to one decimal places
            accuracy = round(accuracy * 100, 1)
            
            result.append(accuracy)
        
        
        # append result to the empty list created
        results.append(result)

    # convert the results to tabular form using Pandas Dataframe function
    results = pd.DataFrame(
        results,
        columns=[
            "Test Size",
            "Accuracy_LR",
            "Accuracy_KNN",
            "Accuracy_RAND",
            "Accuracy_DTREE",
            "Accuracy_LDA",
            "Accuracy_NB"
            # Accuracy_model is the accuracy of each particular model
        ], 
    )
    return results
 
    
def preprocessing(data):
    """Convert categorical column to continuous"""
    
    le = LabelEncoder()
    data["Class"] = le.fit_transform(data["Class"])
    """Split the columns into features and target"""
    features = data.drop(["Class"], axis=1)
    target = data.Class
    return train_test_split(features, target, test_size=0.3, random_state=42)


def build_model(model, train_features, test_features, train_target, test_target):
    """ This function takes a model and the train & test features, 
    train & test targets as params and fits the train data into a model . 
    Returns result of the evaluated metrics of the model
    in a tabular form.
    
    Test_size used: 0.3
    Args:
        model: Dictionary of different models
        train_features: 70% of the features in dataset
        test_features: 30% of the features in dataset
        train_target: 70% of the target column (Class)
        test_target: 30% of the target column (Class)
        
    Prints:
        Cross Validation score
        Accuracy Score
        Confusion Matrix
        Classification Report
    """

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
