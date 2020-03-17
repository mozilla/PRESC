import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns

from sklearn.metrics import confusion_matrix,f1_score

def eval_model(clf,x_test,y_test,x_train,y_train):
    """
    Evaluates the trained classfier. Uses the metrics 
    1. Test Set Accuray
    2. Train Set Accuracy
    3. F1 score
    Also, it plots the confusion matrix
    """

    print("Test Accuracy :",clf.score(x_test,y_test)*100,"%")
    print("Train Accuracy :",clf.score(x_train,y_train)*100,"%")
    if len(y_train.unique())==2:
        print("F1 Score :",f1_score(y_test,clf.predict(x_test)))
    plot_confusion_matrix(y_test,clf.predict(x_test))


def plot_confusion_matrix(y_true, y_pred):
    """
    This is a helper function that helps us visualize the true-positives, true-negatives, false-positives and false-negatives. It can also be used for multi-class classification.
    """
    mtx = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)
    plt.ylabel('Label')
    plt.xlabel('Prediction')