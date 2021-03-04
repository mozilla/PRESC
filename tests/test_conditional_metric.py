from copy import deepcopy

import pytest
import yaml

from presc.report.runner import load_config
from presc.evaluations.conditional_metric import (
    ConditionalMetric,
    METRIC,
    ConditionalMetricResult,
)


COLUMN_OVERRIDE_YAML = """
columns:
  a:
    num_bins: 5
"""

COLNAME_LIST_YAML = """
conditional_metric:
  columns_include:
    - a
    - c
    - e
"""


@pytest.fixture
def config_default():
    return load_config()


@pytest.fixture
def config_col_override(config_default):
    conf = deepcopy(config_default)
    extra = yaml.load(COLUMN_OVERRIDE_YAML, Loader=yaml.FullLoader)
    conf["evaluations"]["conditional_metric"]["computation"]["columns"] = extra[
        "columns"
    ]
    return conf


@pytest.fixture
def config_colname(config_col_override):
    conf = deepcopy(config_col_override)
    extra = yaml.load(COLNAME_LIST_YAML, Loader=yaml.FullLoader)
    conf["evaluations"]["conditional_metric"]["columns_include"] = extra[
        "conditional_metric"
    ]["columns_include"]
    return conf


@pytest.fixture
def result_class_no_display(monkeypatch):
    class CMRPatched(ConditionalMetricResult):
        def display_result(self, xlab, ylab):
            print(f"{xlab}:{len(self.vals)}")

    from presc.evaluations import conditional_metric

    monkeypatch.setattr(conditional_metric, "ConditionalMetricResult", CMRPatched)


def test_eval_compute_for_column(
    test_dataset, classification_model, config_default, config_col_override
):
    # Defaults
    cme = ConditionalMetric(classification_model, test_dataset, config_default)
    cmr = cme.compute_for_column("a", metric=METRIC)
    assert len(cmr.vals) == 10
    assert cmr.num_bins == 10

    # Column-specific override
    cme = ConditionalMetric(classification_model, test_dataset, config_col_override)
    cmr_a = cme.compute_for_column("a", metric=METRIC)
    assert len(cmr_a.vals) == 5
    assert cmr_a.num_bins == 5
    cmr_b = cme.compute_for_column("b", metric=METRIC)
    assert len(cmr_b.vals) == 10
    assert cmr_b.num_bins == 10

    # kwarg override
    cmr_a = cme.compute_for_column("a", metric=METRIC, num_bins=4, quantile=True)
    assert len(cmr_a.vals) == 4
    assert cmr_a.num_bins == 4
    assert cmr_a.quantile is True
    cmr_b = cme.compute_for_column("b", metric=METRIC, num_bins=4)
    assert len(cmr_b.vals) == 4
    assert cmr_b.num_bins == 4


def test_eval_display(
    test_dataset,
    classification_model,
    config_default,
    config_colname,
    result_class_no_display,
    capsys,
):
    cme = ConditionalMetric(classification_model, test_dataset, config_default)
    cme.display()
    cols_displayed = capsys.readouterr().out
    assert len(cols_displayed.split()) == 5

    cme.display(colnames=["a", "c"])
    cols_displayed = capsys.readouterr().out
    assert [x.strip() for x in cols_displayed.split()] == ["a:10", "c:3"]

    cme = ConditionalMetric(classification_model, test_dataset, config_colname)
    cme.display()
    cols_displayed = capsys.readouterr().out
    assert [x.strip() for x in cols_displayed.split()] == ["a:5", "c:3", "e:10"]
