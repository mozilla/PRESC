import random
from threading import Thread
from sklearn.pipeline import Pipeline


class SyntheticDataStreamer(Thread):
    """Generate a stream of synthetic data sampling batches continuously.

    This function generates batches of synthetic data using the sampler and
    parameters specified in a ClassifierCopy instance, and continuously adds
    them to the specified queue as soon as there are any slots available. The
    batches of data can then be consumed by several concurrent threads or any
    process that requires them. In particular, they can also be used by
    ContinuousCopy instances.

    Attributes
    ----------
        classifier_copy : presc.copies.copying.ClassifierCopy
            An instance of the ClassifierCopy class with the original,
            classifier, the copy classifier, and its associated sampling method
            of choice including the sampling parameters.
        data_stream : queue.Queue
            The queue where the continuous synthetic data generator will store
            the synthetic data batches to share them with the main thread.
        verbose : bool
            If set to True a note when stopping the streamer is printed.
    """

    def __init__(self, classifier_copy, data_stream, verbose=False):
        Thread.__init__(self)
        self.classifier_copy = classifier_copy
        self.data_stream = data_stream
        self.verbose = verbose
        if classifier_copy.random_state is None:
            self.random_state = random.randint(0, 1000000)
        else:
            self.random_state = classifier_copy.random_state

    def run(self):
        self.data_streamer_run = True

        while self.data_streamer_run:
            data_block = self.classifier_copy.generate_synthetic_data(
                random_state=self.random_state
            )
            self.random_state = self.random_state + 1
            self.data_stream.put(data_block)

    def stop(self):
        self.data_streamer_run = False
        if self.verbose:
            print("Stopping data streamer...\n")



class ContinuousCopy(Thread):
    """Continuous classifier copy for online classifiers.

    This class allows to instantiate online classifiers to perform sequential
    incremental training of the classifier copy, which can be carried out
    continuously non-stop. It keeps taking the necessary data batches from the
    specified queue as soon as they are available, but it is agnostic as to how
    the data is added to the queue.

    Attributes
    ----------
        classifier_copy : presc.copies.copying.ClassifierCopy
            An instance of the ClassifierCopy class with the original,
            classifier, its associated sampling method of choice, and the online
            copy classifier, which needs to be capable of incremental training.
        data_stream : queue.Queue
            The queue where the batches of data for the incremental training of
            the online copy can be found.
        fit_kwargs : dict or dict of dicts
            A dictionary with the fit parameters needed for the online copy
            classifier, or for each element of the online copy pipeline. In the
            latter case, there must be an entry for each element of the pipeline
            and each one in turn must have a dictionary with its fit parameters.
        verbose : bool
            If True it provides details at each iteration of the number of
            samples used so far for the incremental training of the copy, and
            evaluates the performance of the resulting copy.
        test_data : presc.dataset.Dataset
            Dataset with the test data that will be used to evaluate the copy
            after each training iteration.
    """

    def __init__(
        self, classifier_copy, data_stream, fit_kwargs={}, verbose=False, test_data=None
    ):
        Thread.__init__(self)
        self.classifier_copy = classifier_copy
        self.data_stream = data_stream
        self.fit_kwargs = fit_kwargs
        self.verbose = verbose
        self.test_data = test_data

        self.iterations = 0
        self.n_samples = 0

        if isinstance(self.classifier_copy.copy, Pipeline):
            self.is_pipeline = True
        else:
            self.is_pipeline = False

    def run(self):
        """Function to start the copying of the continuous online classifier."""
        self.continuous_copy_run = True

        self.partial_fit_ok = check_partial_fit(self.classifier_copy.copy)
        if not self.partial_fit_ok:
            print("Sorry, not all those transformers/estimators have 'partial_fit'.")

        while self.continuous_copy_run and self.partial_fit_ok:
            # Get data
            data_block = self.data_stream.get()
            n_samples_block = len(data_block.df)

            # If pipeline, train elements of pipeline sequentially,
            # and use transformed data to fit the next transformer or estimator
            if self.is_pipeline:
                n_estimators = len(self.classifier_copy.copy.named_steps)
                X = data_block.features.copy()
                for index, element in enumerate(self.classifier_copy.copy.named_steps):
                    self.classifier_copy.copy[element].partial_fit(
                        X, y=data_block.labels, **self.fit_kwargs[element]
                    )
                    # Transform data for next estimator except for the last one
                    if index + 1 < n_estimators:
                        X = self.classifier_copy.copy.named_steps[element].transform(X)

            # If single classifier, train model with this block
            else:
                self.classifier_copy.copy.partial_fit(
                    data_block.features, data_block.labels, **self.fit_kwargs
                )
            self.iterations += 1
            self.n_samples += n_samples_block

            if self.verbose:
                print("\nIteration: ", self.iterations)
                print("Samples: ", self.n_samples, "\n")

                self.classifier_copy.evaluation_summary(
                    test_data=self.test_data, synthetic_data=data_block
                )

    def stop(self):
        """Function to stop the continuous copying of the online classifier."""
        self.continuous_copy_run = False
        if self.verbose:
            print("Stopping online classifier copier...\n")
            print(f"The classifier copy trained for {self.iterations} iterations")
            print(f"with a total of {self.n_samples:,} samples.\n".replace(",", "."))


def check_partial_fit(estimator_pipeline):
    """Function to check if estimator or pipeline have partial_fit.

    This function checks if an estimator or classifier have a partial_fit
    function implemented and thus can be used for incremental fits and
    ContinuousCopy instances. In the case of sklearn pipelines, it checks wether
    each and every one of the elements of the pipeline has the same function.

    Parameters
    ----------
    estimator_pipeline : sklearn-type classifier or sklearn.pipeline.Pipeline
        The estimator or pipeline that we want to check.

    Returns
    -------
    bool
        Returns True if the classifier or the pipeline can be fitted
        incrementally.
    """
    partial_fit_ok = True
    if isinstance(estimator_pipeline, Pipeline):
        # Check that all pipeline elements have "partial_fit"
        for element in estimator_pipeline.named_steps:
            if "partial_fit" not in dir(estimator_pipeline.named_steps[element]):
                partial_fit_ok = False
                return partial_fit_ok
    else:
        if "partial_fit" not in dir(estimator_pipeline):
            partial_fit_ok = False
    return partial_fit_ok
