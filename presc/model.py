def is_misclassified(y_predicted, y_true):
    """Return a boolean Series indicating which test cases were misclassified."""
    return y_predicted != y_true


class ClassificationModel:
    """Represents a classification problem.

    Instances wrap a ML model together with its associated dataset.

    Args:
        classifier (sklearn Classifier): the classifier to wrap
        dataset (Dataset): the data associated with the classification model
        should_train (bool): should the classifier be (re-)trained on the given dataset?
    """

    def __init__(self, classifier, dataset, should_train=False):
        self._classifier = classifier
        self._dataset = dataset
        if should_train:
            # Train the classifier on the given dataset.
            self.train()
        # Precompute predictions for the test set.
        self._test_preds = self.predict()
        # Precompute a series indicating misclassified test samples.
        self._test_misclassified = is_misclassified(
            self._test_preds, self._dataset.test_labels
        )

    def train(self, X_train=None, y_train=None):
        """Train the underlying classification model.

        Training data defaults to the instance training set.
        """
        X_train = X_train or self._dataset.train_features
        y_train = y_train or self._dataset.train_labels
        self._classifier.fit(X_train, y_train)

    def predict(self, X_test=None):
        """Predict labels for the given feature dataset.

        Defaults to the instance test set.
        """
        X_test = X_test or self._dataset.test_features
        return self._classifier.predict(X_test)

    @property
    def test_predictions(self):
        """Returns the predicted labels for the test set."""
        return self._test_preds

    @property
    def test_misclassified(self):
        """Returns a boolean Series indicating misclassified test set cases."""
        return self._test_misclassified

    @property
    def dataset(self):
        """Returns the underlying dataset."""
        return self._dataset
