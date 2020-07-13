import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from classification_models import Classifier
from data_processing import convert_binary

def calibration_plot(models=None, X_train=None, y_train=None, X_test=None, y_test=None, group=[(0, 3), (1, 2)], labels=[0, 1]):
    """Function to plot the calibration curve for different binary classifiers."""
    if models is None:
        print('Models not specified.')
        return
    elif 'mlp' in models:
        print('MLP not supported.')
        return
    elif X_train is None or y_train is None or X_test is None or y_test is None:
        print('Data not provided.')
        return

    allowed_models = ['logistic', 'svm', 'knn']
    
    if len(models) < 2:
        models = allowed_models

    y_train = convert_binary(y_train, group=group, labels=labels)
    y_test = convert_binary(y_test, group=group, labels=labels)

    if y_train is None or y_test is None:
        return
    
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly Calibrated")
    
    for model in models:
        classifier = Classifier(model=model)
        classifier.train(X_train, y_train)
        classifier.validate(X_test, y_test)
        
        probabilities = classifier.prediction_probabilities()[:, 1]
        fraction_positives, mean_predicted = calibration_curve(y_test, probabilities, n_bins=10)

        print('Model: {0}, Accuracy: {1}'.format(model, classifier.model_accuracy()))

        plt.plot(mean_predicted, fraction_positives, label=model)

    plt.ylabel("Fraction of Positives")
    plt.xlabel("Mean Predicted Probability")
    plt.title("Calibration Plot")
    plt.legend()
    plt.show()