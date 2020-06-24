from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

def calibration(clf, x_train, y_train, x_val, y_val):
    '''
    The function plots the calibration curves for classifaction models.

    Parameters:
        clf : trained classification moodel
        x_train : array-like, shape(n_train_samples, n_features)
        y_train : of length n_train_samples
        x_val : array-like, shape(n_val_samples, n_features)
        y_val : of length of n_val_samples

    Returns:
        null

    '''

    methods = ['sigmoid', 'isotonic']
    
    fop = {}
    apv ={}
    clf_score = {}
    for i in range(len(methods)):

        calibrated_model = CalibratedClassifierCV(clf, method=methods[i], cv=5)
        calibrated_model.fit(x_train, y_train)

        y_score = calibrated_model.predict_proba(x_val)[:,1]
        fop[i], apv[i] = calibration_curve(y_val, y_score, n_bins = 10, normalize=True)

        clf_score[i] = brier_score_loss(y_val, y_score, pos_label=1)

    plt.figure(figsize=(10,6))
    plt.plot([0,1],[0,1])
    plt.plot(apv[0], fop[0], label='Sigmoid (Brier loss={:.3f})'.format(clf_score[0]))
    plt.plot(apv[1], fop[1], label='Isotonic(Brier loss={:.3f})'.format(clf_score[1]))
    plt.grid()
    plt.xlabel("Average Probability")
    plt.ylabel("Fraction of Positive")
    plt.title("Calibration Plots")
    plt.legend()
    plt.show()