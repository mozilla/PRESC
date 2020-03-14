import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
def misclassificationGraphs(y_test, preds):
    '''
    Plots a graph to show default values
    Agrs: The label y
    '''
    predsSeries = pd.Series(preds)
    predsSeries.reindex(index=y_test.index)
    df_miss = pd.concat([y_test, predsSeries], axis=1, join='inner')
    df_miss = df_miss.rename(columns={"default payment next month": "y_test", 0: "predsSeries"})
    ax1 = sns.countplot(x="predsSeries", hue="y_test", data=df_miss, palette="Blues")
 #   plt.show()
    
def plotGrapthForAttributes(df):
    '''
    Plots Graphs for each attibute for EDA
    Args: The dataframe
    '''
    # Creating a new dataframe with categorical variables
    subset = df[['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 
                   'PAY_5', 'PAY_6', 'default payment next month']]

    f, axes = plt.subplots(3, 3, figsize=(20, 15), facecolor='white')
    f.suptitle('FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)')
    ax1 = sns.countplot(x="SEX", hue="default payment next month", data=subset, palette="Blues", ax=axes[0,0])
    ax2 = sns.countplot(x="EDUCATION", hue="default payment next month", data=subset, palette="Blues",ax=axes[0,1])
    ax3 = sns.countplot(x="MARRIAGE", hue="default payment next month", data=subset, palette="Blues",ax=axes[0,2])
    ax4 = sns.countplot(x="PAY_0", hue="default payment next month", data=subset, palette="Blues", ax=axes[1,0])
    ax5 = sns.countplot(x="PAY_2", hue="default payment next month", data=subset, palette="Blues", ax=axes[1,1])
    ax6 = sns.countplot(x="PAY_3", hue="default payment next month", data=subset, palette="Blues", ax=axes[1,2])
    ax7 = sns.countplot(x="PAY_4", hue="default payment next month", data=subset, palette="Blues", ax=axes[2,0])
    ax8 = sns.countplot(x="PAY_5", hue="default payment next month", data=subset, palette="Blues", ax=axes[2,1])
    ax9 = sns.countplot(x="PAY_6", hue="default payment next month", data=subset, palette="Blues", ax=axes[2,2]);
