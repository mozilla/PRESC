from copy import deepcopy

import pytest
import yaml
from numpy.testing import assert_array_equal
from pandas import MultiIndex

from presc.evaluations.conditional_distribution import (
    ConditionalDistribution,
    ConditionalDistributionResult,
)


COLUMN_OVERRIDE_YAML = """
columns:
  a:
    common_bins: False
"""

COLNAME_LIST_YAML = """
conditional_distribution:
  columns_include:
    - a
    - c
    - e
"""


@pytest.fixture
def config_col_override(config_default):
    conf = deepcopy(config_default)
    extra = yaml.load(COLUMN_OVERRIDE_YAML, Loader=yaml.FullLoader)
    conf["evaluations"]["conditional_distribution"]["computation"]["columns"] = extra[
        "columns"
    ]
    return conf


@pytest.fixture
def config_colname(config_col_override):
    conf = deepcopy(config_col_override)
    extra = yaml.load(COLNAME_LIST_YAML, Loader=yaml.FullLoader)
    conf["evaluations"]["conditional_distribution"]["columns_include"] = extra[
        "conditional_distribution"
    ]["columns_include"]
    return conf


@pytest.fixture
def result_class_no_display(monkeypatch):
    class CDRPatched(ConditionalDistributionResult):
        def display_result(self, xlab):
            print(f"{xlab}:{isinstance(self.bins.index, MultiIndex)}")

    from presc.evaluations import conditional_distribution

    monkeypatch.setattr(
        conditional_distribution, "ConditionalDistributionResult", CDRPatched
    )


def test_eval_compute_for_column(
    test_dataset, classification_model, config_default, config_col_override
):
    # Defaults
    cde = ConditionalDistribution(classification_model, test_dataset, config_default)
    cdr = cde.compute_for_column("a")
    # Same number of bins in each group
    assert_array_equal(cdr.vals.groupby(["label", "predicted"]).size().unique(), 3)
    assert cdr.binning == "fd"
    assert cdr.common_bins is True

    # Column-specific override
    cde = ConditionalDistribution(
        classification_model, test_dataset, config_col_override
    )
    cdr_a = cde.compute_for_column("a")
    assert cdr_a.vals.groupby(["label", "predicted"]).size().nunique() > 1
    assert cdr_a.common_bins is False
    cdr_b = cde.compute_for_column("b")
    assert cdr_b.vals.groupby(["label", "predicted"]).size().nunique() == 1
    assert cdr_b.common_bins is True

    # kwarg override
    cdr_a = cde.compute_for_column("a", binning=3, common_bins=False)
    assert_array_equal(cdr_a.vals.groupby(["label", "predicted"]).size().unique(), 3)
    # Bins are not all the same
    assert isinstance(cdr_a.bins.index, MultiIndex)
    assert cdr_a.binning == 3
    assert cdr_a.common_bins is False
    cdr_b = cde.compute_for_column("b", binning=3)
    assert_array_equal(cdr_b.vals.groupby(["label", "predicted"]).size().unique(), 3)
    assert not isinstance(cdr_b.bins.index, MultiIndex)
    assert cdr_b.binning == 3
    assert cdr_b.common_bins is True


def test_eval_display(
    test_dataset,
    classification_model,
    config_default,
    config_colname,
    result_class_no_display,
    capsys,
):
    cde = ConditionalDistribution(classification_model, test_dataset, config_default)
    cde.display()
    cols_displayed = capsys.readouterr().out
    assert len(cols_displayed.split()) == 5

    cde.display(colnames=["a", "c"])
    cols_displayed = capsys.readouterr().out
    assert [x.strip() for x in cols_displayed.split()] == ["a:False", "c:False"]

    cde = ConditionalDistribution(classification_model, test_dataset, config_colname)
    cde.display()
    cols_displayed = capsys.readouterr().out
    assert [x.strip() for x in cols_displayed.split()] == [
        "a:True",
        "c:False",
        "e:False",
    ]
