.PHONY: upload pytest setup_conda

all: pytest

setup_conda:
	# Install all dependencies and setup repo in dev mode
	conda env create -f environment.yml

pytest:
	pytest
	flake8 presc tests

pytest_ci:
	pytest -sv
	flake8 presc tests
	$(MAKE) -C sphinx_docs clean html

# build:
# 	bin/create_version
# 	docker build -t ${IMAGE_NAME} .

upload:
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
