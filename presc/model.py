from pandas import Series, DataFrame


class ClassificationModel:
    """Represents a classification problem.

    Instances wrap a ML model together with its associated training dataset.

    Args:
        classifier (sklearn Classifier): the classifier to wrap
        dataset (Dataset): optionally include the associated training dataset
        should_train (bool): should the classifier be (re-)trained on the given dataset?
    """

    def __init__(self, classifier, train_dataset=None, should_train=False):
        self._classifier = classifier
        self._train_dataset = train_dataset
        if should_train:
            # Train the classifier on the given dataset.
            self.train()

    def train(self, train_dataset=None):
        """Train the underlying classification model.

        train_dataset: a Dataset to train on. Defaults to the prespecified
            training dataset, if any.
        """
        if train_dataset is None:
            train_dataset = self._train_dataset

        self._classifier.fit(train_dataset.features, train_dataset.labels)

    def predict_labels(self, test_dataset):
        """Predict labels for the given test dataset.

        Returns a like-indexed Series.
        """
        pred = self._classifier.predict(test_dataset.features)
        return Series(pred, index=test_dataset.features.index)

    def predict_probs(self, test_dataset):
        """Compute predicted probabilities for the given test dataset.

        Fails if not supported by the classifier.

        Returns a like-indexed DataFrame of probabilities for each class.
        """
        pred = self._classifier.predict_proba(test_dataset.features)
        return DataFrame(pred, index=test_dataset.features.index)

    @property
    def classifier(self):
        """Returns the underlying classifier."""
        return self._classifier
