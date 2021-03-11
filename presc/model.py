from pandas import Series, DataFrame

from presc.utils import PrescError


class ClassificationModel:
    """Represents a classification problem.

    Instances wrap a ML model together with its associated training dataset.

    Args:
        classifier (sklearn Classifier): the classifier to wrap
        dataset (Dataset): optionally include the associated training dataset
        retrain_now (bool): should the classifier first be (re-)trained on the given dataset?
    """

    def __init__(self, classifier, train_dataset=None, retrain_now=False):
        self._classifier = classifier
        self._train_dataset = train_dataset
        if retrain_now:
            # Train the classifier on the given dataset.
            self.train()

    def train(self, train_dataset=None):
        """Train the underlying classification model.

        Parameters
        ----------
        train_dataset : presc.dataset.Dataset
            A Dataset to train on. Defaults to the pre-specified training dataset, if any.
        """
        if train_dataset is None:
            train_dataset = self._train_dataset

        self._classifier.fit(train_dataset.features, train_dataset.labels)

    def predict_labels(self, test_dataset):
        """
        Predict labels for the given test dataset.

        Parameters
        ----------
        test_dataset : presc.dataset.Dataset

        Returns
        -------
        Series
            A like-indexed Series.
        """
        pred = self._classifier.predict(test_dataset.features)
        return Series(pred, index=test_dataset.features.index)

    def predict_probs(self, test_dataset):
        """
        Compute predicted probabilities for the given test dataset.

        This must be supported by the underlying classifier, otherwise an
        error will be raised.

        Parameters
        ----------
        test_dataset : presc.dataset.Dataset

        Returns
        -------
        DataFrame
            A like-indexed DataFrame of probabilities for each class.
        """
        try:
            pred = self._classifier.predict_proba(test_dataset.features)
            return DataFrame(pred, index=test_dataset.features.index)
        except AttributeError as e:
            raise PrescError(
                "classifier does not support predicted probabilities"
            ) from e

    @property
    def classifier(self):
        """Returns the underlying classifier."""
        return self._classifier
