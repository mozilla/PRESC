from confuse.core import RootView
from presc import config


class LocalConfig(RootView):
    """Configuration view that overrides but doesn't modify another view.

    This is useful for temporarily overriding options, eg. with feature-specific
    settings, while still taking advantage of the confuse resolution and
    templating functionalities.

    The override is dynamic, so it will always pull in the most recent values of
    the underlying configuration.

    from_config: a confuse Configuration (RootView) instance to override. If
    not provided, defaults to the global PRESC config.
    """

    def __init__(self, from_config=None):
        super().__init__([])
        if not from_config:
            from_config = config.settings
        self._source_config = from_config
        self._local_sources = []

    def set(self, value):
        # Convert value to appropriate format and insert at the front of
        # self.sources.
        super().set(value)
        # Move the result to the local sources list
        self._local_sources.insert(0, self.sources.pop(0))

    def resolve(self):
        # Resolve against the current state of the underlying config.
        self._refresh_sources()
        return super().resolve()

    def _refresh_sources(self):
        # Pull in the latest state of the source config.
        try:
            # If the source config is also a LocalConfig
            self._source_config._refresh_sources()
        except AttributeError:
            pass
        self.sources = self._local_sources + self._source_config.sources
