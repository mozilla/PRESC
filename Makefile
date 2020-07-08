.PHONY: build up tests flake8 setup_conda

setup_conda:
	# Install all dependencies and setup repo in dev mode
	conda env create -f environment.yml
	python setup.py develop

pytest:
	pytest
	flake8 presc tests scripts analysis

black:
	black presc tests scripts analysis

build:
	bin/create_version
	docker build -t ${IMAGE_NAME} .

upload:
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
