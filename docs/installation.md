# Installation

PRESC can be installed using pip by running:

```shell
$ pip install presc
```

## Development install

The source code for PRESC is available on
[GitHub](https://github.com/mozilla/PRESC).
Once you have cloned the repository, you can set up the development environment
using `conda`.

For this you need to have Conda installed (we recommend the
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)
distribution).

To set up the environment, run the following from the root directory of the
cloned repository.
This will also enable a pre-commit hook to verify that code conforms to flake8
and black formatting rules.
On Windows, these commands should be run from the Anaconda command prompt.

```shell
$ conda env create -f environment.yml
$ conda activate presc
$ python setup.py develop
$ pre-commit install
```

Once the environment is set up, you should be able to run the tests:

```shell
$ pytest
```

