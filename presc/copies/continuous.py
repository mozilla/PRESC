from threading import Thread
from sklearn.pipeline import Pipeline


class SyntheticDataStreamer(Thread):
    """Generate a stream of synthetic data continuously sampling batches."""

    def __init__(self, classifier_copy, data_stream, verbose=False):
        Thread.__init__(self)
        self.classifier_copy = classifier_copy
        self.data_stream = data_stream
        self.verbose = verbose
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


def check_partial_fit(estimator_pipeline):
    """Function to check if estimator or pipeline have partial_fit."""
    partial_fit_ok = True
    if isinstance(estimator_pipeline, Pipeline):
        # Check that all pipeline elements have "partial_fit"
        for element in estimator_pipeline.named_steps:
            if "partial_fit" not in dir(estimator_pipeline.named_steps[element]):
                partial_fit_ok = False
                return partial_fit_ok
    else:
        if "partial_fit" not in dir(estimator_pipeline.copy):
            partial_fit_ok = False
    return partial_fit_ok
