.PHONY: upload pytest black setup_conda

setup_conda:
	# Install all dependencies and setup repo in dev mode
	conda env create -f environment.yml

pytest:
	pytest
	flake8 presc tests

black:
	black presc tests

# build:
# 	bin/create_version
# 	docker build -t ${IMAGE_NAME} .

upload:
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
