import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def plot_curve(classifiers, X, y, bins=10):

    """Draw calibration plots for the given classifiers. All the classifiers will be
  trained first and then the calibration points are calculated based on their predicted
  values on test dataset. Those calibration points are then plotted.

    Parameters
    ----------
    classifiers : array, shape (n_classifiers, 2)
        Object instances of multiple binary classifiers.
    X : array, shape (n_samples, n_feature_columns)
        Training Data with n_samples and n_feature_columns
    y : array, shape (n_samples,)
        True Labels
    bins : int
        Number of bins. 
    
    References
    ----------
    sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
  """

    fig, ax = plt.subplots()
    fig.suptitle("Calibration plots with data transformation")

    # Axis labels
    ax.set_xlabel("Predicted Probabilities")
    ax.set_ylabel("Actual Probabilities")

    # Add y_true line
    line = mlines.Line2D([0, 1], [0, 1], color="black")
    transform = ax.transAxes

    line.set_transform(transform)
    ax.add_line(line)

    # Intialize StandardScaler
    scaler_sc = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values.ravel(), test_size=0.3, random_state=1
    )

    # Data Transform
    scaler_sc.fit(X_train)
    X_train = scaler_sc.transform(X_train)
    X_test = scaler_sc.transform(X_test)

    predictions = []

    # Loop through each classifier, train and predict
    for elem in classifiers:
        clf = elem[1]
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        predictions.append([elem[0], y_test, y_pred])
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred[:, 1], n_bins=bins
        )
        plt.plot(
            fraction_of_positives,
            mean_predicted_value,
            marker="o",
            linewidth=1,
            label=elem[0],
        )

    plt.legend()
    plt.show()

    return


def plot_curve_without_transformation(classifiers, X, y, bins=10):

    """Draw calibration plots for the given classifiers. All the classifiers will be
  trained first and then the calibration points are calculated based on their predicted
  values on test dataset. Those calibration points are then plotted.

    Parameters
    ----------
    classifiers : array, shape (n_classifiers, 2)
        Object instances of multiple binary classifiers.
    X : array, shape (n_samples, n_feature_columns)
        Training Data with n_samples and n_feature_columns
    y : array, shape (n_samples,)
        True Labels
    bins : int
        Number of bins. 
    
    References
    ----------
    sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
  """

    fig, ax = plt.subplots()
    fig.suptitle("Calibration plots with data transformation")

    # Axis labels
    ax.set_xlabel("Predicted Probabilities")
    ax.set_ylabel("Actual Probabilities")

    # Add y_true line
    line = mlines.Line2D([0, 1], [0, 1], color="black")
    transform = ax.transAxes

    line.set_transform(transform)
    ax.add_line(line)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values.ravel(), test_size=0.3, random_state=1
    )

    predictions = []

    # Loop through each classifier, train and predict
    for elem in classifiers:
        clf = elem[1]
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        predictions.append([elem[0], y_test, y_pred])
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred[:, 1], n_bins=bins
        )
        plt.plot(
            fraction_of_positives,
            mean_predicted_value,
            marker="o",
            linewidth=1,
            label=elem[0],
        )

    plt.legend()
    plt.show()

    return
