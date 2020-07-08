# Performance Robustness Evaluation for Statistical Classifiers

[![CircleCI](https://circleci.com/gh/mozilla/PRESC.svg?style=svg)](https://circleci.com/gh/mozilla/PRESC)
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

Code formatting guidelines should strinctly adhere  to [Python Black](https://pypi.org/project/black/) formatting guidelines. Please ensure that all PRs pass a local black formatting check.




## Information for Outreachy participants

__Please note that this project is currently closed to new Outreachy
contributions.__

- At this time, we are only considering Outreachy candidates who have submitted
  a PR on or before _Friday March 20_.
- If you have submitted a PR by this date, you may continue working on existing
  PRs or create new ones as usual. All your contributions will be considered.
- If you have not yet submitted a PR by this date, we will unfortunately not be
  able to consider you as an Outreachy candidate for this round.


This project is intentionally broadly scoped, and the initial phase will be
  exploratory.

- The goal is to propose and test out ideas related to the evaluation of
  classifiers, rather than jumping straight into building features.
- Tasks are open-ended and can be worked on by multiple
  contributors at the same time, as different people may propose
  complimentary ideas or approaches.

You can ask for help and discuss your ideas on [gitter](https://gitter.im/PRESC-outreachy/community).

### Issues

Tasks are managed using the [GitHub issue tracker](https://github.com/mozilla/PRESC/issues).

- As issues represent general exploratory tasks at this point, they will
  generally not be assigned to a single person.
- If you want to work on a task, drop a comment in the issue.
- You are welcome to make a contribution to an issue even if others are
  already working on it. You may also expand on someone else's work, eg.
  testing out the methodology with different datasets or models.
- As the project matures, we may start filing targeted issues, eg. to fix
  specific bugs, which will get assigned to specific person
- You are also welcome to contribute your own issues if there is a direction you
  would like to explore relating to the project focus.

### Contributions

Contributions can be made by submitting a [pull request](https://help.github.com/articles/using-pull-requests) against this repository. Learn more about [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

- We ask each Outreachy participant to make a contribution completing
  [#2](https://github.com/mozilla/PRESC/issues/2) (train and test a
  classification model). This will help you to become familiar with machine
  learning and the tools if you are not already. Please submit as a PR following
  the [guidelines](#contribution-guidelines) above.
    * This task __must__ be completed in order to be considered as an intern on
      this project
- If you would like initial feedback on your contribution before it is ready for
  submission, you may open a PR with "WIP:" at the start of the title and
  request review. This tag ('work in progress') indicates that the PR is not
  ready to be merged. When it is ready for final submission, you can modify the
  title to remove the "WIP:" tag.
- Should you use a separate jupyter notebook for comparing different models? If
  you had a PR merged in to satisfy issue #2 already and are now comparing
  models for another issue, then a new notebook would be helpful. That being
  said, a notebook should satisfy the following criteria:

    a) it should run beginning to end without error

    b) it should be easy to follow and have a clear narrative presenting context,
   data, results, and interpretation. This may mean some redundancy in code, but
   will often mean that your notebook is much more helpful to other people
   looking at it in isolation (including reviewers).


## Getting started

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/miniconda.html).

2. Fork this repository and clone it into your local machine(using git CLI).

3. Setup and activate environment. Note that this will also enable a
   pre-commit hook to verify that code conforms to flake8 and black
   formatting rules:

```
 $ conda env create -f environment.yml
 $ conda activate presc
 $ pre-commit install
```


__For Windows:__ Open anaconda prompt and `cd` into the folder where you cloned the repository

```
cd PRESC
```
then type the above commands to activate the environment.


4. Run Jupyter. The notebook will open in your browser at `localhost:8888` by default.

```
 $ jupyter notebook
```
After running this commands you will see the notebook containing the datasets and now you can start working with it.

We recommend everyone start by working on
[#2](https://github.com/mozilla/PRESC/issues/2).


### Getting started with GitHub

The git/GitHub open-source workflow can be rather confusing if you haven't used
it before. To make a contribution to the project, the general steps you need to
take are:

- Install [git](https://git-scm.com/downloads) on your computer
- Fork the repo on Github (ie. make your own personal copy)
- Clone your fork to your local computer
- Set remote origin (https://github.com/<_user_>/PRESC.git) and upstream (https://github.com/mozilla/PRESC.git)
- Create a new branch for every issue or new work that you do.
(To avoid merge conflicts keep your work in a separate folder in the same branch if it contains more than a few files.)
- Commit changes locally on your computer
- Push your changes to your fork on Github
- Submit a pull request (PR) against the main repo from your fork.

A few commands to start with everytime you work with a GIT repository:
- `git fetch upstream master`
- `git checkout FETCH_HEAD -b <new_branch_name>`
- Make changes
- `git status`
This will show the files that have been modified, deleted or created
- `git add .` (To add all the modified files)
	OR
  `git add <file_name>` (To add a specific file)
- `git commit -m '<commit_message>'`
- `git push`
If you get an error message on executing the above command, enter the suggested `git push` command.

Now, click on the link that you see once the `push` command is executed to create a Pull Request. While creating a Pull Request do mention `[ Fixes: #<issue_number> ]` in the description. This will link the issue to the Pull Request for which the latter is created.

Once your Pull Request is merged do `git pull --rebase upstream master`. This will update your fork with local changes and the ones made from upstream. This is to ensure there are no file conflicts.

Here are some resources to learn more about parts of this workflow you are
unfamiliar with:

- [GitHub Guides](https://guides.github.com/)
    * In particular, the [git handbook](https://guides.github.com/introduction/git-handbook/) explains some of the basics of the version control system
    * There is a [page](https://guides.github.com/activities/forking/)
      explaining the forking/pull request workflow you will be using to
      contribute.
- The [Git Book](https://git-scm.com/book/en/v2) is much more detailed but a good reference
    * The [Getting Started](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control) section is worth reading
    * There are also some [videos](https://git-scm.com/videos) on getting set up
- This [repo](https://github.com/aSquare14/Git-Cheat-Sheet) by a previous
  Outreachy contributor lists many other resources and tutorials.
- This [video tutorial series](https://www.youtube.com/playlist?list=PL6gx4Cwl9DGAKWClAD_iKpNC0bGHxGhcx) on Youtube may also be helpful
- [First Contributions](https://github.com/firstcontributions/first-contributions#first-contributions) is a good place to actually practise and put your understanding to test. Feel free to make mistakes as you go along learning to make your first contribution. 

Feel free to reach out to the mentors by email or on Gitter if you have further
questions or are having trouble getting set up!


## Resources


- [This](https://github.com/brandon-rhodes/pycon-pandas-tutorial) is a great tutorial to learn Pandas.
- [Tutorial](https://www.youtube.com/watch?v=HW29067qVWk) on Jupyter Notebook.
- The [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
  is a good place to start learning the scikit-learn library as well as machine
  learning methodology and comes with lots of examples.
- [This](https://builtin.com/data-science/supervised-machine-learning-classification) page has a nice overview of classification models.
