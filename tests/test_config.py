import presc
from presc.configuration import PrescConfig

from confuse import Configuration
import yaml


CONFIG_OVERRIDE_FILE = """
report:
    title: ABC
evaluations:
  conditional_metric:
    computation:
      num_bins: 3
"""


def test_package_config():
    # Check that package-wide config has been initialized and some expected
    # default settings are present.
    assert isinstance(presc.config, PrescConfig)
    assert isinstance(presc.config.settings, Configuration)
    assert presc.config.settings["report"]["title"].get()
    assert isinstance(
        presc.config.settings["evaluations"]["conditional_metric"]["computation"].get(),
        dict,
    )


def test_presc_config(tmp_path):
    conf = PrescConfig()

    # Default settings have been loaded.
    # This is raise an error if the value is not defined in the config
    default_title = conf.settings["report"]["title"].get()
    default_cm_comp = conf.settings["evaluations"]["conditional_metric"][
        "computation"
    ].get()

    assert default_title is not None
    assert isinstance(default_cm_comp, dict)

    # Overriding settings with another config file
    override_file = tmp_path / "config_override.yaml"
    with open(override_file, "w") as f:
        f.write(CONFIG_OVERRIDE_FILE)

    conf.update_from_file(override_file)
    assert (
        conf.settings["evaluations"]["conditional_metric"]["computation"][
            "num_bins"
        ].get(int)
        == 3
    )
    assert conf.settings["report"]["title"].get(str) == "ABC"
    assert (
        conf.settings["evaluations"]["conditional_metric"]["computation"]
        != default_cm_comp
    )

    # Overriding settings manually.
    conf.set({"report": {"title": "ABCDEF", "author": "Xyz"}})
    assert conf.settings["report"]["title"].get() == "ABCDEF"
    assert conf.settings["report"]["author"].get() == "Xyz"
    conf.set(
        {
            "evaluations.conditional_metric.computation.num_bins": 7,
            "evaluations.conditional_metric.columns_include": ["a", "b"],
            "evaluations.conditional_metric.computation.columns.a.num_bins": 5,
        }
    )
    assert (
        conf.settings["evaluations"]["conditional_metric"]["computation"][
            "num_bins"
        ].get(int)
        == 7
    )
    assert conf.settings["evaluations"]["conditional_metric"]["columns_include"].get(
        list
    ) == ["a", "b"]
    assert (
        conf.settings["evaluations"]["conditional_metric"]["computation"]["columns"][
            "a"
        ]["num_bins"].get(int)
        == 5
    )

    # Reverting back to defaults
    conf.reset_defaults()
    assert conf.settings["report"]["title"].get() == default_title
    assert (
        conf.settings["evaluations"]["conditional_metric"]["computation"].get()
        == default_cm_comp
    )

    # dump() should give valid YAML. Parsing should not fail.
    yaml.load(conf.dump(), Loader=yaml.FullLoader)
