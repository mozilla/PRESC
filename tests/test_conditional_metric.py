import pytest
import yaml

from presc.evaluations.conditional_metric import (
    ConditionalMetric,
    METRIC,
    ConditionalMetricResult,
)
from presc import global_config
from presc.configuration import PrescConfig


COLUMN_OVERRIDE_YAML = """
computation:
  columns:
    a:
      num_bins: 5
"""

COLNAME_LIST_YAML = """
columns_include:
  - a
  - c
  - e
"""


@pytest.fixture
def config_col_override():
    return yaml.load(COLUMN_OVERRIDE_YAML, Loader=yaml.FullLoader)


@pytest.fixture
def config_colname_and_override():
    return yaml.load(COLNAME_LIST_YAML + COLUMN_OVERRIDE_YAML, Loader=yaml.FullLoader)


@pytest.fixture
def result_class_no_display(monkeypatch):
    class CMRPatched(ConditionalMetricResult):
        def display_result(self, xlab, ylab):
            print(f"{xlab}:{len(self.vals)}")

    from presc.evaluations import conditional_metric

    monkeypatch.setattr(conditional_metric, "ConditionalMetricResult", CMRPatched)


def test_eval_compute_for_column(
    test_dataset, classification_model, config_col_override
):
    # Defaults
    cme = ConditionalMetric(classification_model, test_dataset)
    cmr = cme.compute_for_column("a", metric=METRIC)
    assert len(cmr.vals) == 10
    assert cmr.num_bins == 10

    # Global override
    global_config.set({"evaluations.conditional_metric.computation.num_bins": 6})
    cme = ConditionalMetric(classification_model, test_dataset)
    cmr = cme.compute_for_column("a", metric=METRIC)
    assert len(cmr.vals) == 6
    assert cmr.num_bins == 6

    # Evaluation-level override:
    # explicit settings
    global_config.reset_defaults()
    conf_default = global_config.dump()
    cme = ConditionalMetric(
        classification_model, test_dataset, settings={"computation.num_bins": 7}
    )
    cmr = cme.compute_for_column("a", metric=METRIC)
    assert len(cmr.vals) == 7
    assert cmr.num_bins == 7
    assert global_config.dump() == conf_default
    # config object + settings
    new_config = PrescConfig(global_config)
    new_config.set({"evaluations.conditional_metric.computation.num_bins": 4})
    cme = ConditionalMetric(
        classification_model,
        test_dataset,
        settings={"computation.quantile": True},
        config=new_config,
    )
    cmr = cme.compute_for_column("a", metric=METRIC)
    assert len(cmr.vals) == 4
    assert cmr.num_bins == 4
    assert cmr.quantile is True
    assert global_config.dump() == conf_default

    # Column-specific override
    cme = ConditionalMetric(
        classification_model, test_dataset, settings=config_col_override
    )
    cmr_a = cme.compute_for_column("a", metric=METRIC)
    assert len(cmr_a.vals) == 5
    assert cmr_a.num_bins == 5
    cmr_b = cme.compute_for_column("b", metric=METRIC)
    assert len(cmr_b.vals) == 10
    assert cmr_b.num_bins == 10
    assert global_config.dump() == conf_default

    # kwarg override
    conf_cme = cme._config.dump()
    cmr_a = cme.compute_for_column("a", metric=METRIC, num_bins=4, quantile=True)
    assert len(cmr_a.vals) == 4
    assert cmr_a.num_bins == 4
    assert cmr_a.quantile is True
    cmr_b = cme.compute_for_column("b", metric=METRIC, num_bins=3)
    assert len(cmr_b.vals) == 3
    assert cmr_b.num_bins == 3
    assert global_config.dump() == conf_default
    assert cme._config.dump() == conf_cme


def test_eval_display(
    test_dataset,
    classification_model,
    config_colname_and_override,
    result_class_no_display,
    capsys,
):
    cme = ConditionalMetric(classification_model, test_dataset)
    cme.display()
    cols_displayed = capsys.readouterr().out
    assert len(cols_displayed.split()) == 5

    cme.display(colnames=["a", "c"])
    cols_displayed = capsys.readouterr().out
    assert [x.strip() for x in cols_displayed.split()] == ["a:10", "c:3"]

    cme = ConditionalMetric(
        classification_model, test_dataset, settings=config_colname_and_override
    )
    cme.display()
    cols_displayed = capsys.readouterr().out
    assert [x.strip() for x in cols_displayed.split()] == ["a:5", "c:3", "e:10"]
