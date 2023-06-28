FROM continuumio/miniconda3:4.12.0 as build-env
# Singularity builds from docker use tini, but raises warnings
# we set it up here correctly for singularity
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

### EXAMPLES ###

### Build dev branch
# sudo docker build --network=host -t saber:master .

### Build test branch
# sudo docker build --build-arg git_branch=test --network=host -t saber:test .

################

Workdir /opt

### Definitions:

ENV PYTHONPATH=/opt/:/opt/libs
ENV CONDA_ENV saber_cenv
ENV ENV_FILE environment_relaxed.yml
# ENV ENV_FILE environment.yml

# haven't figured out local build yet
ENV git_branch master
# ARG git_branch=master

## Set up Conda:
COPY $ENV_FILE /opt/$ENV_FILE
#RUN conda update -n base -c defaults conda
RUN conda env create -f /opt/$ENV_FILE
# this bypasses conda shenanigans by just making the env available globally
ENV PATH=/opt/conda/envs/for_container/bin:$PATH
# Install SABer and Dependencies:
RUN pip3 install git+https://github.com/hallamlab/SABer.git@${git_branch}#egg=SABerML

# move to clean execution environment to reduce image size
# jammy is 22.04 long term support
FROM ubuntu:jammy
ENV CONDA_ENV saber_cenv
COPY --from=build-env /opt/conda/envs/$CONDA_ENV /opt/conda/envs/$CONDA_ENV
COPY --from=build-env /tini /tini
# this bypasses conda shenanigans by just making the env available globally
ENV PATH=/opt/conda/envs/for_container/bin:$PATH

# ### Install apt dependencies
# RUN DEBIAN_FRONTEND=noninteractive apt-get update \
#                     && apt-get install --no-install-recommends -y \
#                        wget libgmp3-dev \
#                     && apt-get clean \
#                     && rm -rf /var/lib/apt/lists/*

## We do some umask munging to avoid having to use chmod later on,
## as it is painfully slow on large directores in Docker.
RUN old_umask=`umask` && \
    umask 0000 && \
    umask $old_umask

## Make things work for Singularity by relaxing the permissions:
#RUN chmod -R 755 /opt

# singularity doesn't use the -s flag for tini, and that causes warnings
ENTRYPOINT ["/tini", "-s", "--"]
