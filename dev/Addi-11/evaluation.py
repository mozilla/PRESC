from sklearn.metrics import accuracy_score, confusion_matrix,average_precision_score,precision_recall_curve, plot_precision_recall_curve, f1_score

from sklearn.utils import class_weight
from dataloader import train_test_split_data

X_train, X_test,y_train, y_test = train_test_split_data()

def predict(classifier):
    return classifier.predict(X_test)

def evaluate(classifier):
    y_score = predict(classifier)
    accuracy = accuracy_score(y_test, y_score)
    f_score = f1_score(y_test, y_score)
    disp = plot_precision_recall_curve(classifier, X_test, y_test)

    print("\nAccuracy : ",accuracy)
    print("F1 score : ",f_score)

