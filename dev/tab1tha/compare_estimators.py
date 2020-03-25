"""This program seeks to compare the prediction results of two or more machine learning models, 
when they are trained and used on the same dataset. It is the first step in guiding the choice of 
an optimal model for the dataset of interest. """

import k_nn
import s_v_m
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

filename = "vehicles.csv"
df = load_dataset(filename)

data = df.drop("Class", axis=1)
target = df["Class"]

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, stratify=target, random_state=42)
"""scaling improves model performance significantly however it raised a questionable TypeError
data_train = StandardScaler(data_train)
data_test = StandardScaler(data_test)"""
acc_knn, target_pred_knn = k_nn.k_nearest(data_train=data_train, target_train=target_train, data_test=data_test, target_test=target_test)
acc_svc, target_pred_svc = s_v_m.s_vee_c(data_train, data_test,target_train, target_test)
"""The K-Nearest neighbor and support vector classifier models are trained and tested with tuned hyperparameters.
Their accuracy scores and prediction values are extracted from their respective functions. """

for a in [target_pred_knn, target_pred_svc]:
    a = list(a)

#target_pred_knn = list(target_pred_knn)
#target_pred_svc = list(target_pred_svc)
conf_matrix_knn = confusion_matrix(target_test, target_pred_knn)   
conf_matrix_svc = confusion_matrix(target_test, target_pred_svc)   

combi_dict = {"Ref":list(target_test)}
compare_df = pd.DataFrame(combi_dict, columns=["Ref"])
compare_df["KNN"] = target_pred_knn
compare_df["SVC"] = target_pred_svc

#print(compare_df)
print(compare_df.Ref.value_counts())
print(compare_df.KNN.value_counts())
print(compare_df.SVC.value_counts())

#confu_matrix_display(conf_matrix_knn, target_vals)
#sns.pairplot(df)
#plt.show()
