import seaborn as sns
import matplotlib.pyplot as plt

def plotxx(cnfmatrix):
    fig= plt.figure(figsize=(6,3))
    # to plot the graph
    sns.heatmap(cnfmatrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    return

