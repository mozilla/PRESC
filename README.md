# PRESC: Performance and Robustness Evaluation for Statistical Classifiers

[![CircleCI](https://circleci.com/gh/mozilla/PRESC.svg?style=svg)](https://circleci.com/gh/mozilla/PRESC)
[![Join the chat at https://gitter.im/PRESC-outreachy/community](https://badges.gitter.im/PRESC-outreachy/community.svg)](https://gitter.im/PRESC-outreachy/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

PRESC is a toolkit for the evaluation of machine learning classification
models.
Its goal is to provide insights into model performance which extend beyond
standard scalar accuracy-based measures and into areas which tend to be
overlooked in application, including:

- Generalizability of the model to unseen data for which the training set may
  not be representative
- Sensitivity to statistical error and methodological choices
- Performance evaluation localized to meaningful subsets of the feature space
- In-depth analysis of misclassifications and their distribution in the feature
  space

More details about the specific features we are considering are presented in the
[project roadmap](./docs/ROADMAP.md).
We believe that these evaluations are essential for developing confidence in
the selection and tuning of machine learning models intended to address user
needs, and are important prerequisites towards building
[trustworthy AI](https://foundation.mozilla.org/en/internet-health/trustworthy-artificial-intelligence/).

As a tool, PRESC is intended for use by ML engineers to assist in the
development and updating of models.
It will be usable in the following ways:

- As a standalone tool which produces a graphical report evaluating a given
  model and dataset
- As a Python package/API which can be integrated into an existing pipeline
- As a step in a Continuous Integration workflow: evaluations run as a part of
  CI, for example, on regular model updates, and fail if metrics produce
  unacceptable values.

We are using the standard Python scientific stack (numpy/pandas/jupyter).
In order to streamline development while the project is still in its early
stages, we are restricting focus to
[scikit-learn](https://scikit-learn.org/stable/index.html)
supervised classification models, and we are prototyping report visualizations
in [Jupyter notebooks](./examples).
For the time being, the following are considered __out of scope__:

- Models built in machine learning frameworks other than scikit-learn
- User-facing evaluations, eg. explanations
- Evaluations which depend explicitly on domain context or value judgements of
  features, eg. protected demographic attributes. A domain expert could use
  PRESC to study misclassifications across such protected groups, say, but the
  PRESC evaluations themselves should be agnostic to such determinations.
- Analyses which do not involve the model, eg. class imbalance in the training
  data

There is a considerable body of recent academic research addressing these
topics, as well as a number of open-source projects solving related problems.
Where possible, we plan to offer integration with existing tools which align
with our vision and goals.

This project was the subject of an [Outreachy](https://www.outreachy.org/)
internship during Summer 2020.
Submissions from the Spring 2020 application period have been archived in this
[this repo](https://github.com/mozilla/PRESC-Outreachy-archive) in their
original state, and will be integrated here as needed.


## Notes for contributors

Contributions are welcome.
We are using the repo [issues](https://github.com/mozilla/PRESC/issues) to
manage project tasks in alignment with the [roadmap](./docs/ROADMAP.md), as well
as hosting discussions.
You can also reach out on [Gitter](https://gitter.im/PRESC-outreachy/community).

We recommend that submissions for new feature implementations include a Juypter
notebook demonstrating their application to a real-world dataset and model.
The repo includes a few well-known [datasets](./datasets) for testing.
If you wish to propose another, please make sure to clearly indicate its source
and license (if applicable).

This repo adheres to [Python black](https://pypi.org/project/black/)
formatting, which is enforced by a pre-commit hook (see below).


## Getting started

Make sure you have conda (eg. [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
installed. `conda init` should be run during installation to set the PATH
properly.

Set up and activate the environment. This will also enable a pre-commit hook to
verify that code conforms to flake8 and black formatting rules.
On Windows, these commands should be run from the Anaconda command prompt.

```shell
$ conda env create -f environment.yml
$ conda activate presc
$ python setup.py develop
$ pre-commit install
```

To run tests:

```shell
$ pytest
```


## Updating PyPI

Use `twine` to upload a new version to PyPI.

```shell
$ python setup.py sdist
$ twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```

The Makefile target `make upload` will invoke `twine` for you.
