# declare var across stages
ARG CONDA_ENV=saber_cenv

# FROM continuumio/miniconda3:4.12.0 as build-env
FROM condaforge/mambaforge as build-env

### EXAMPLES ###

### Build dev branch
# sudo docker build --network=host -t saber:master .

### Build test branch
# sudo docker build --build-arg git_branch=test --network=host -t saber:test .

################

### Definitions:

# scope var from global
ARG CONDA_ENV
ENV ENV_FILE environment.yml

### Install apt dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN apt-get install --no-install-recommends -y \
    python3-dev gcc libc-dev libffi-dev libgmp3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

### Set up Conda:
COPY ./$ENV_FILE /opt/$ENV_FILE
RUN mamba env create --no-default-packages -f /opt/$ENV_FILE \
    && mamba clean -afy

# --------------------------------------------------------
# move to clean execution environment to reduce image size
# jammy is 22.04 long term support
FROM ubuntu:jammy

# scope var from global
ARG CONDA_ENV
COPY --from=build-env /opt/conda/envs/$CONDA_ENV /opt/conda/envs/$CONDA_ENV
# bypass conda activate shenanigans by just making the env available globally
ENV PATH=/opt/conda/envs/$CONDA_ENV/bin:$PATH

# move over source code, setup entry point, and register to PATH
COPY ./src /app/src
COPY ./entry.sh /app/saber
RUN chmod +x /app/saber
ENV PATH=/app:$PATH

# Singularity builds from docker use tini, but raises warnings
# we set it up here correctly for singularity
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

## We do some umask munging to avoid having to use chmod later on,
## as it is painfully slow on large directores in Docker.
RUN old_umask=`umask` && \
    umask 0000 && \
    umask $old_umask

# singularity doesn't use the -s flag for tini, and that causes warnings
ENTRYPOINT ["/tini", "-s", "--"]
