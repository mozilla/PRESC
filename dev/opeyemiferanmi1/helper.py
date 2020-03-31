# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from sklearn.metrics import roc_curve


def data_description(data):
    """Displays some basic descriptions of the imported data"""
    print('Shape of the data:' + str(data.shape), "\n")
    print("\n")
    print('Value count of each vehicle class:'+'\n', data.Class.value_counts())
    print('Any Missing values?', data.isnull().sum().any())
    print("\n")
    print('Data type of each column'+'\n', data.dtypes)


def corr_plot(data):
    data.corr()['Class'].sort_values().plot\
        (kind='bar', figsize=(18, 6))


def positive_corr_plot(data_normal):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    # plot on 0 row and 0 column
    sns.barplot(x="ELONGATEDNESS",y="Class", data=data_normal, ax=ax[0, 0])
    sns.barplot(x="HOLLOWS_RATIO",y="Class", data=data_normal, ax=ax[0, 1])
    sns.barplot(x="MAX.LENGTH_ASPECT_RATIO",y="Class", data=data_normal, ax=ax[1, 0])
    sns.barplot(x="SKEWNESS_ABOUT_MINOR",y="Class", data=data_normal, ax=ax[1, 1])


def scale_fit_transform_data(df, column_list):
    """Takes in a dataframe and a list of column names to transform
        returns a dataframe of scaled values"""
    df_to_scale = df[column_list]
    x = df_to_scale.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    df_to_scale = pd.DataFrame(x_scaled, columns=df_to_scale.columns)
    return df_to_scale


def spot_check_algorithm(x_train, y_train, n_splits=10, random_state=42):
    """Spot checks different algorithms showing accuracy
     mean and standard deviation"""
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVC', SVC(gamma='auto')))
    models.append(('RFC', RandomForestClassifier()))
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LDA()))
    # evaluate each model in turn
    results = []
    names = []
    details = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold)
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def split(x, y):
    """Split the train and test data"""
    """The splitting of this dataset into folds is governed by criteria(stratify) 
    such that each fold has the same proportion/distribution of 
    observations with the categorical value of the Class column."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test


def train_LDA(x_train, y_train, x_test, y_test):
    """Train model with LDA"""
    lda_model = LDA()
    lda_model.fit(x_train, y_train)
    y_pred = pd.DataFrame(lda_model.predict(x_test))
    return y_pred


def print_results(y_test, y_pred):
    """Print Model Accuracy"""
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy:", accuracy * 100)
    print(classification_report(y_pred, y_test))


def plot_confusion_matrix(y_test, y_pred):
    """ plots Confusion Matrix and Misclassified vehicle classes """
    labels = ["van", "saab", "bus", "opel"]
    cm = confusion_matrix(y_test, y_pred, labels=None)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.ylabel("Actual Vehicle Class")
    plt.xlabel("Predicted Vehicle Class")
    for label in df_cm.columns:
        df_cm.at[label, label] = 0
    ax = df_cm.plot(kind="bar", title="Misclassified Vehicle Classes")
    ax.set_xlabel("Vehicle Classes")
    ax.set_ylabel("Number of Incorrectly Predicted Class")
    plt.show()


def van(x):
    if x == 'van':
        return 1
    else:
        return 0


def saab(x):
    if x == 'saab':
        return 1
    else:
        return 0


def bus(x):
    if x == 'bus':
        return 1
    else:
        return 0


def opel(x):
    if x == 'opel':
        return 1
    else:
        return 0


def plot_roc_curve(y_pred_loss,y_test_loss):
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test_loss, y_pred_loss)
    # plot no class
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC curve")
    # show the plot
    plt.show()
