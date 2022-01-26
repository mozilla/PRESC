from setuptools import find_packages, setup

VERSION = "0.3.0"

with open("README.md", "r") as fh:
    # Keep only the top part until the first section break.
    # (temporary workaround)
    readme = fh.read()
    long_description = readme[: readme.find("\n\n## ")]

setup(
    name="presc",
    use_scm_version=False,
    version=VERSION,
    setup_requires=["setuptools_scm", "pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*"]),
    description="Performance Robustness Evaluation for Statistical Classifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mozilla Corporation",
    author_email="dzeber@mozilla.com",
    url="https://github.com/mozilla/PRESC",
    license="MPL 2.0",
    python_requires=">=3.7",
    install_requires=[
        "confuse",
        "numpy",
        "pandas>=1.0.0",
        "scikit-learn>=0.23.1",
        "scipy",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "jupyter-book>=0.9.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment :: Mozilla",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    zip_safe=False,
)
