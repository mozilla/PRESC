import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import modules as modle
from sklearn.metrics import classification_report, confusion_matrix

def Missclasssification_visualization(y_test, y_predict,data):
    
    labels = data["Class_code"].unique()
    label = data["Class"].unique()
    cmatrix = pd.DataFrame(
            data= modle.model_confusion_matrix(y_test, y_predict, data) , 
            columns=label,
            index=label
            )

    ##Because we only want to print missclassification, 
    for label in cmatrix.columns:
            cmatrix.at[label, label] = 0

    graph = cmatrix.plot(kind="bar", title="Visualization of Missclassification", stacked=True)
    graph.set_xlabel("Class Labels")
    graph.set_ylabel("Missclassified Classes")
    plt.show()
    plt.show()