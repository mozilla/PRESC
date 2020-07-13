import matplotlib.pyplot as plt
import seaborn as sns


def raw(df):
    """This function provides insights into the
     structure of content in the dataset. It shows that target values are all clearly defined 
     and discrete. Hence, a classification model is appropriate. """
    print("*" * 50 + "HEAD" + "*" * 50)
    print(df.head())
    print("*" * 50 + "TAIL" + "*" * 50)
    print(df.tail())
    print("*" * 50 + "DESCRIBE" + "*" * 50)
    print(df.describe())
    print("*" * 50 + "INFO" + "*" * 50)
    print(df.info())
    print("*" * 100)


def graph_visualize(df, col):
    """This function plots a pair plot that provides a detailed view of interrelationships in
     the dataset """
    # visualisation
    plt.figure()
    sns.pairplot(df, hue=col)
    plt.show()


def violin_visualize(df,col="Class"):
    """This function investigates the distribution of observation results 
    among the target values by drawing a violin plot"""
    # create a figure instance
    fig = plt.figure()
    """create an axes instance and specifies appropriate coordinates.
     Values are just defaults chosen through tinkering. """
    ax = fig.add_axes([0, 0, 1, 1])
    # create the boxplot
    bp = ax.violinplot(df[col])
    plt.show(bp)

def histo(df, col):
    """This function plots a histogram to show the behaviour of a feature. Its recieves a 
    pandas dataframe and its column of interest as parameters. A matplotlib plot is returned. """
    sns.countplot(x= col, data=df)
    plt.title('Spread plot')
    plt.show()
