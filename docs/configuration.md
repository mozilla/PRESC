# Configuration

The PRESC evaluations support configuration at a number of levels.
Configuration settings are managed by `presc.configuration.PrescConfig`
instances.

The persistent global configuration is available as `global_config`:

```python
from presc import global_config

# Display the current config in YAML format
print(global_config.dump())
```

## Report settings

Currently available config settings for the report are as follows:

```yaml
# Look and structure of the report
report:
  # Report title (passed to jupyter-book config)
  title: PRESC report
  # Report author (passed to jupyter-book config)
  author: ""
  # List of evaluations to show in the report. Each is added in a separate page.
  # Values must correspond to module names for available evaluations.
  # The report will include only those listed in `evaluations_include`,
  # after removing any listed in `evaluations_exclude`.
  # "*" means 'all available evaluations'.
  evaluations_include: "*"
  evaluations_exclude: null
```

These can set globally by passing a dict of option values to the
`global_config`:

```python
global_config.set({"report.title": "My Report"})
```

Overrides to the report settings can also be passed to the `ReportRunner` in a
YAML file. For example, if the file `myconf.yml` contains

```yaml
report:
  title: Another report
```

this can be set using

```python
report = ReportRunner("./my_output_dir", config_filepath="myconf.yml")
```

Overrides can also be passed in at runtime using:

```python
report.run(cm, test_dataset, settings={"report.title": "My new report"})
```

Instance- or method-level overrides like these do not change the global config.
However, if global config options are changed, these changes will also be
reflected in local configs (unless they are already overridden). For example:

```python
global_config.set({"report.author": "Me"})
# This will pull in the updated author string.
report.run(cm, test_dataset, settings={"report.title": "My new report"})
```

The goal of this flexibility is to make it easy to experiment with different
settings while developing a report or exploring evaluations.

## Include/exclude

Include/exclude settings are available to control which evaluation methods are
included in the report, and which parts of an evaluation are computed. They
operate by restricting to listed include values, and then removing listed
exclude values. The special values `"*"` and `None` are interpreted as "all" or
"none", respectively.

The evaluations included in a report can be specified by listing module names,
eg.

```yaml
evaluations_include:
  - conditional_metric
  - conditional_distribution
```

to include only these two, or 

```yaml
evaluations_exclude:
  - conditional_metric
```

to exclude this one. These can be passed either in a YAML file or a dict, as
described above.


## Evaluation settings

These are discussed in more detail in the Evaluations section.
Evaluations provide similar local override functionality.
