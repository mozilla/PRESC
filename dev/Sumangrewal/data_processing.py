import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

dataset = pd.read_csv("winequality.csv")



x=dataset.iloc[:,:-2].values
y=dataset.iloc[:,12].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
accs=accuracy_score(y_test,y_pred)

print(cm)
print(accs)

