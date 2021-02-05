import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.exceptions import NotFittedError
from numpy import arange
from pandas import Series, DataFrame

from presc.model import ClassificationModel
from presc.dataset import Dataset

TEST_SET_SELECTOR = arange(100) >= 80


@pytest.fixture
def in_test_set():
    rows = arange(100)
    return (rows < 10) | (rows >= 90)


@pytest.fixture
def train_dataset(dataset_df, in_test_set):
    return Dataset(dataset_df[~in_test_set], label_col="label")


@pytest.fixture
def test_dataset(dataset_df, in_test_set):
    return Dataset(dataset_df[in_test_set], label_col="label")


def pipeline_is_trained(pl):
    # Quick check that the classifier has fitted params
    return hasattr(pl.named_steps["clf"], "coef_")


@pytest.fixture
def pipeline_classifier():
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), make_column_selector(dtype_include="number")),
            ("encoder", OneHotEncoder(), make_column_selector(dtype_include=object)),
        ],
        remainder="passthrough",
    )
    return Pipeline([("scaler", ct), ("clf", LogisticRegression())])


def test_classification_model(train_dataset, test_dataset, pipeline_classifier):
    cm = ClassificationModel(pipeline_classifier)
    assert cm.classifier is pipeline_classifier
    assert not pipeline_is_trained(cm.classifier)
    with pytest.raises(NotFittedError):
        cm.predict_labels(test_dataset)

    cm.train(train_dataset)
    assert pipeline_is_trained(cm.classifier)

    yp = cm.predict_labels(test_dataset)
    assert isinstance(yp, Series)
    assert (yp.index == list(range(10)) + list(range(90, 100))).all()
    assert (yp != test_dataset.labels).sum() == 5

    pp = cm.predict_probs(test_dataset)
    assert isinstance(pp, DataFrame)
    assert (pp.index == list(range(10)) + list(range(90, 100))).all()
    assert pp.values.max() <= 1
    assert pp.values.min() >= 0
    assert (pp.loc[:, 1].round() != test_dataset.labels).sum() == 5


def test_wrapped_dataset(train_dataset, pipeline_classifier, test_dataset):
    cm = ClassificationModel(pipeline_classifier, train_dataset)
    assert not pipeline_is_trained(cm.classifier)
    with pytest.raises(NotFittedError):
        cm.predict_labels(test_dataset)

    cm.train()
    assert pipeline_is_trained(cm.classifier)
    # Prediction should work without error
    cm.predict_labels(test_dataset)

    cm = ClassificationModel(pipeline_classifier, train_dataset, should_train=True)
    assert pipeline_is_trained(cm.classifier)
    # Prediction should work without error
    cm.predict_labels(test_dataset)
