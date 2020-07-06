FROM continuumio/miniconda3
ENV PYTHONDONTWRITEBYTECODE 1

MAINTAINER Muriel Rovira Esteva <mroviraesteva@mozilla.com>
# add a non-privileged user for installing and running
# the application
RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid 10001 --home /app --create-home app 

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gettext curl \
                                               libopenblas-dev libatlas3-base gfortran && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# First copy requirements.txt so we can take advantage of docker
# caching.
COPY . /app

RUN make setup_conda

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate presc && \
    python setup.py install

#USER app

# Using /bin/bash as the entrypoint works around some volume mount issues on Windows
# where volume-mounted files do not have execute bits set.
# https://github.com/docker/compose/issues/2301#issuecomment-154450785 has additional background.
ENTRYPOINT ["/bin/bash", "/app/bin/run"]
