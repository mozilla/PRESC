"""Config management for PRESC, handled using confuse.Configuration."""

from pathlib import Path

from confuse import Configuration

from presc.utils import PrescError

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_default.yaml")


class LocalConfig(Configuration):
    """
    Confuse config view that overrides but doesn't modify another view.

    This is useful for temporarily overriding options, eg. with feature-specific
    settings, while still taking advantage of the confuse resolution and
    templating functionalities.

    The override is dynamic, so it will always pull in the most recent values of
    the underlying configuration.

    from_config: a confuse Configuration (RootView) instance to override.
    """

    def __init__(self, from_config):
        super().__init__("PRESC_local", read=False)
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


class PrescConfig:
    """
    Wrapper around a confuse Configuration object.

    This is used for managing config options in PRESC, including the global config.

    Attributes
    ----------
    from_config : PrescConfig
        A PrescConfig instance to override. If None, the config is initialized to the default settings.
    """

    def __init__(self, from_config=None):
        if from_config:
            self._config = LocalConfig(from_config.settings)
        else:
            self._config = Configuration("PRESC", read=False)
            self.reset_defaults()

    def reset_defaults(self):
        """Reset all options to their defaults."""
        self._config.clear()
        self.update_from_file(DEFAULT_CONFIG_PATH)

    def update_from_file(self, file_path):
        """Override current settings with those in the given YAML file."""
        self._config.set_file(str(file_path))

    def set(self, settings):
        """Update one or more config options.

        These should be specified in a dict, either mirroring the nested
        structure of the configuration file, or as flat key-value pairs using
        dots to indicate nested namespaces.

        Examples
        --------
        ``config.set({"report": {"title": "My Report", "author": "Me"}})``
        ``config.set({"report.title": "My Report", "report.author": "Me"})``

        """
        if not isinstance(settings, dict):
            raise PrescError("Config settings must be specified in a dict")
        self._config.set_args(settings, dots=True)

    @property
    def settings(self):
        """Access the underlying confuse object."""
        return self._config

    def dump(self):
        """Dump the current config in YAML format."""
        return self._config.dump()

    # Make option access work on the PrescConfig:
    def __getitem__(self, key):
        return self._config.__getitem__(key)

    def get(self, template=None):
        # If template is None, defer to the underlying default arg.
        template_arg = {}
        if template:
            template_arg["template"] = template
        return self._config.get(**template_arg)

    def flatten(self):
        return self._config.flatten()


# The global config object:
global_config = PrescConfig()
