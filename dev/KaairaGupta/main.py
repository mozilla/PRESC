from Ml_functions import svm
from Ml_functions import KNeighbors
from Ml_functions import LogisticR
from Ml_functions import DecisionT
from Ml_functions import RandomF

k_accuracy, k_precision, k_recall = KNeighbors()
lr_accuracy, lr_precision, lr_recall = LogisticR()
dt_accuracy, dt_precision, dt_recall = DecisionT()
rf_accuracy, rf_precision, rf_recall = RandomF()

print("Kneighbors Accuracy = " , k_accuracy , " Kneighbors Precision = " , k_precision , " Kneighbors Recall = " ,k_recall)
print("LogisticRegression Accuracy = " , lr_accuracy , " LogisticRegression Precision = " , lr_precision , " LogisticRegression Recall = " , lr_recall)
print("DecisionTreeClassifier Accuracy = " , dt_accuracy , " DecisionTreeClassifier Precision = " , dt_precision , ", DecisionTreeClassifier Recall = " , dt_recall)
print("RandomForestClassifier Accuracy = " , rf_accuracy , " RandomForestClassifier Precision = " , rf_precision , ", RandomForestClassifier Recall = " , rf_recall)