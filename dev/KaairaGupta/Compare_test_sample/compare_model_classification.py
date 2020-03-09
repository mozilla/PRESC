import matplotlib.pyplot as plt
import numpy as np

def compare_model_classification(model_1,model_2):
    """
        This function develops a visualization to compare predicted class probabilities across models for binary classifiers.
        Input: takes input of 2 arrays, one for each model's test samples prediction. The arrays are of the size (n*3), where n are the number of outputs. The 3 columns are of the form (P(y=0), P(y=1), actual value of y)
        Output: Visualisation in the form of plots for each class.

    """
    t = [],[]
    
    for i in range(len(model_1)):
        y = model_1[i][2]
        temp = [min(model_1[i][y],round(model_1[i][y])),min(model_2[i][y],round(model_2[i][y]))]
        if not (temp[0]==0 and temp[1]==0):
            t[y].append(temp)

    t_trans = [np.array(t[0]).transpose(), np.array(t[1]).transpose()]
    
    for i in range(2):
        x = np.arange(len(t[i]))
        plt.bar(x-0.125, t_trans[i][:][0], color='y', width=0.25, label="Model 1")
        plt.bar(x-0.125+0.25, t_trans[i][:][1], color='g', width=0.25, label="Model 2")
        title = "Performace for y = {} for two models.".format(i)
        plt.title(title)
        plt.xticks(x)
        plt.legend()
        plt.show()
