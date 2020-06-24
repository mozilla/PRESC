from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines





def Calibrated_Curve(models, X_train, y_train, X_test, y_test):
    
    """
    This function is plotting the line graph for the predictive probablities     
    
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for model in models:
        predict = model[1].fit(X_train, y_train).predict_proba(X_test)
        y, x = calibration_curve(y_test, predict[:, 1], n_bins=10)
        plt.plot(x, y, marker="o", linewidth=1, label=model[0])

    calibrated_line = mlines.Line2D([0, 1], [0, 1], label= 'Ideal Calibrated Curve', color="black")
    ax.add_line(calibrated_line)
    fig.suptitle("Calibration plot for WineQuality DataSet")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual probability in each bin")
    plt.legend()
    plt.show()
