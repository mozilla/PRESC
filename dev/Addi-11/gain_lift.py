""" smt """
from dataloader import train_test_split_data
import matplotlib.pyplot as plt
import scikitplot as skplt



def gain_lift_chart(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)
    skplt.metrics.plot_cumulative_gain(y_test, y_score)
    plt.show()


