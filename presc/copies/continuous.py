from sklearn.pipeline import Pipeline


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
