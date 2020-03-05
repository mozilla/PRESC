# Performance Robustness Evaluation for Statistical Classifiers

[![Join the chat at https://gitter.im/PRESC-outreachy/community](https://badges.gitter.im/PRESC-outreachy/community.svg)](https://gitter.im/PRESC-outreachy/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Overview

Mozilla is planning to invest more substantially in privacy-preserving machine
learning models for applications such as recommending personalized content and
detecting malicious behaviour.
As such solutions move towards production, it is essential for us to have
confidence in the selection of the model and its parameters for a particular
dataset, as well as an accurate view into how it will perform in new instances
or as the training data evolves.

While the literature contains a broad array of models, evaluation techniques,
and metrics, their choice in practice is often guided by convention or
convenience, and their sensitivity to different datasets is not always
well-understood.
Additionally, to our knowledge there is no existing software tool that provides
a comprehensive report on the performance of a given model under consideration.

The eventual goal of this project is to build a standard set of tools that
Mozilla can use to evaluate the performance of machine learning models in
various contexts on the basis of the following __principles__:

- Holistic view of model performance
    * Eg. evaluating a model in terms of multiple metrics (ie as a battery of
      tests), and understanding the tradeoffs between them
    * For classification, separating predicted class membership scores from the
      decision rule assigning a final class
- Stability of performance metrics as an additional metric to optimize for
    * Eg. the variability of performance metrics across difference splits of the
      dataset
- Generalizability of the model to unseen data
- Explainability/parsimony when possible
- Reproducibility of results
    * Eg. looking for ways to leverage existing public datasets in the
      evaluation process when the actual data cannot be made public
- Informative failures
    * We can learn a lot by studying the cases where the model performs poorly,
      eg. misclassifications or cases near the decision boundary
    * Failures may have implications for generalizability, appropriateness of
      the choice of model, ethical considerations/bias
    * Eg. do we see systematic failures with a homogeneous distribution within
      themselves but differing from training data.

At this early stage, the focus is on implementing evaluation methodologies in
line with these principles, and testing them out across different models and
datasets.


## Notes for contributors

- We are currently restricting scope to classification supervised learning
  models (both binary and multiclass).
- Tests and analyses should use the datasets provided [in the repo](./datasets).
  Please do not include external datasets in your contributions at this point.
- We are working in Python using the standard data science stack
  (Numpy/Pandas/Scikit-learn/Jupyter).
- Your code should run in the provided [Conda environment](environment.yml). If you feel
  you need an external dependency, you may include an update to the environment
  file in your PR.


### Contribution guidelines

To keep the project structure and review process manageable at this initial
stage, please structure your contributions using the following steps:

- Create a directory with your username in the [`dev`](./dev) dir
- Structure your code into one or more Python modules in that directory
    * Code should be well-documented. Each function should include a docstring.
- Include a [Jupyter
  notebook](https://jupyter-notebook.readthedocs.io/en/stable/) that
  demonstrates a run of your code showing
  printed output, a graph, etc.
    * Code cells in the notebook should only call functions defined in your
      modules. Please do not include any actual code logic in the notebook
      itself.
    * The notebooks should be well-documented. Ideally, each code cell should
      be preceded by a Markdown cell describing why you have included the code
      cell. It can also include commments on the output generated, eg.
      describing features of a graph. These text cells should be more
      high-level than actual code comments, describing the narrative or thought
      process behind your contribution.

We request that contributions be structured in this way prior to getting
reviewed. If you make subsequent contributions, you can include them in the same
directory and reuse module code, but each contribution should include a separate
demonstration notebook.

If you wish to build on someone else's contribution, you can import code from
their modules into yours. Please do not submit PRs directly modifying code from
other contributions at this point, unless to resolve errors or typos.


## Information for Outreachy participants

- This project is intentionally broadly scoped, and the initial phase will be
  exploratory.
    * The goal is to propose and test out ideas related to the evaluation of
      classifiers, rather than jumping straight into building features.
    * Many of the tasks are open-ended and can be worked on by multiple
      contributors at the same time. This will be made clear in the issue
      description.
- Tasks are managed using the [GitHub issue tracker](https://github.com/mozilla/PRESC/issues).
- Contributions can be made by submitting a [pull
  request](https://help.github.com/articles/using-pull-requests) against this
  repository.
- We ask each Outreachy participant to make a contribution completing
  [#2](https://github.com/mozilla/PRESC/issues/2) (train and test a
  classification model). This will help you to become familiar with machine
  learning and the tools if you are not already. Please submit as a PR following
  the [guidelines](#contribution-guidelines) above.
    * This task __must__ be completed in order to be considered as an intern on
      this project
- You can ask for help and discuss your ideas on
  [gitter](https://gitter.im/PRESC-outreachy/community).
- If you would like initial feedback on your contribution before it is ready for
  submission, you may open a PR with "WIP:" at the start of the title and
  request review. This tag ('work in progress') indicates that the PR is not
  ready to be merged. When it is ready for final submission, you can modify the
  title to remove the "WIP:" tag.


## Getting started

__TODO__


## Resources


__TODO__
