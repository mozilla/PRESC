#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


# In[ ]:


def calibration_curve_plot(model,name,X_test,y_test,y):
    """INPUTS:
    *************
    model - list of the objects of applied classifier for calibration comparison
    
    name - list consisting names of the applied classifiers
    
    X_test,y_test - test set used for calibration
    
    y - target attribute
    ***************
    
    decision function - it takes the test set as input and provides the probabilistic prediction for each sample as output."""
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model,name in zip(model,name): 
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob = model.decision_function(X_test)
            prob = (prob - prob.min()) / (prob.max() - prob.min())

        clf_score = brier_score_loss(y_test, prob, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier Score: %1.3f" % (clf_score))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('comparison of Calibration of classifiers (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

