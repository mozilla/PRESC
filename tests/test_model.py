import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.exceptions import NotFittedError
from numpy import arange
from pandas import Series, DataFrame

from presc.model import ClassificationModel
from presc.dataset import Dataset
from presc.utils import PrescError


@pytest.fixture
def in_test_set():
    # Deterministic train/test split for testing
    # This is used with the synthetic `dataset_df` with 100 rows
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
def dataset_transform():
    return ColumnTransformer(
        [
            ("scaler", StandardScaler(), make_column_selector(dtype_include="number")),
            ("encoder", OneHotEncoder(), make_column_selector(dtype_include=object)),
        ],
        remainder="passthrough",
    )


@pytest.fixture
def pipeline_classifier(dataset_transform):
    return Pipeline([("scaler", dataset_transform), ("clf", LogisticRegression())])


@pytest.fixture
def pipeline_classifier_noprobs(dataset_transform):
    # Example of a classifier that doesn't expose `predict_proba`
    return Pipeline([("scaler", dataset_transform), ("clf", RidgeClassifier())])


def test_classification_model(
    train_dataset, test_dataset, pipeline_classifier, pipeline_classifier_noprobs
):
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

    cm_np = ClassificationModel(pipeline_classifier_noprobs)
    cm_np.train(train_dataset)
    with pytest.raises(PrescError):
        cm_np.predict_probs(test_dataset)


def test_wrapped_dataset(train_dataset, pipeline_classifier, test_dataset):
    cm = ClassificationModel(pipeline_classifier, train_dataset)
    assert not pipeline_is_trained(cm.classifier)
    with pytest.raises(NotFittedError):
        cm.predict_labels(test_dataset)

    cm.train()
    assert pipeline_is_trained(cm.classifier)
    # Prediction should work without error
    cm.predict_labels(test_dataset)

    cm = ClassificationModel(pipeline_classifier, train_dataset, retrain_now=True)
    assert pipeline_is_trained(cm.classifier)
    # Prediction should work without error
    cm.predict_labels(test_dataset)
