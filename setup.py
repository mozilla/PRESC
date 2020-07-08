from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PRESC",
    use_scm_version=False,
    version="0.1.1",
    setup_requires=["setuptools_scm", "pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    packages=find_packages(exclude=["archive", "tests", "tests/*"]),
    description="Performance Robustness Evaluation for Statistical Classifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Mozilla Corporation",
    author_email="mroviraesteva@mozilla.com",
    url="https://github.com/mozilla/presc",
    license="MPL 2.0",
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment :: Mozilla",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    zip_safe=False,
)
