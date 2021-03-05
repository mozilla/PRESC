from pathlib import Path

import confuse

from presc.utils import PrescError

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_default.yaml")


class PrescConfig:
    """Wrapper around a confuse Configuration object."""

    def __init__(self):
        self._config = confuse.Configuration("PRESC", read=False)
        self.reset_defaults()

    def reset_defaults(self):
        """Reset all options to their defaults."""
        self._config.clear()
        self._config.set_file(DEFAULT_CONFIG_PATH)

    def update_from_file(self, file_path):
        """Override current settings with those in the given YAML file."""
        self._config.set_file(str(file_path))

    def set(self, settings):
        """Update one or more config options.

        These should be specified in a dict, either mirroring the nested
        structure of the configuration file, or as flat key-value pairs using
        dots to indicate nested namespaces.

        Eg.
        ```
        config.set({"report": {"title": "My Report", "author": "Me"}})
        config.set({"report.title": "My Report", "report.author": "Me"})
        ```
        """
        if not isinstance(settings, dict):
            raise PrescError("Config setting must be specified in a dict")
        self._config.set_args(settings, dots=True)

    @property
    def settings(self):
        """Access the underlying confuse object."""
        return self._config

    def dump(self):
        """Dump the current config in YAML format."""
        return self._config.dump()


config = PrescConfig()
