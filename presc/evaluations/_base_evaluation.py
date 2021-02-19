class BaseEvaluation:
    """Base class for evaluations.

    Args:
        model: the ClassificationModel instance to evaluate, including the
            classifier and dataset.
        config: dict-like containing config options for the evaluation method
    """

    _default_config = {}

    def __init__(self, model, config=None):
        self._model = model
        # Dataset is bundled in model instance
        self._dataset = model.dataset

        # Update any config options passed to the constructor
        self._config = dict(self._default_config)
        if config:
            self._config.update(config)
