#### Make configuration:

#### Assumptions:
## Conda, Docker, Singularity, Python3.*


## Use Bash as default shell, and in strict mode:
SHELL := /bin/bash
.SHELLFLAGS = -ec

## If the parent env doesn't ste TMPDIR, do it ourselves:
TMPDIR ?= /tmp

## Users can override this variable from the command line,
## to install MP binaries somewhere other than /usr/local,
## if they lack root privileges:
DESTDIR ?= /usr/local

## This makes all recipe lines execute within a shared shell process:
## https://www.gnu.org/software/make/manual/html_node/One-Shell.html#One-Shell
.ONESHELL:

## If a recipe contains an error, delete the target:
## https://www.gnu.org/software/make/manual/html_node/Special-Targets.html#Special-Targets
.DELETE_ON_ERROR:

## This is necessary to make sure that these intermediate files aren't clobbered:
.SECONDARY:

### Local Definitions:
PYTHON ?= python3

### Conda + pip install
install-saberenv: #installs SABer deps with conda + pip
	conda env create -f environment.yml
install-saber: # installs saber with pip
	pip install .

### Container Automation
docker-start:
	sudo systemctl start docker

## If git_branch is empty, it probably means that we are building off of a tagged version.
## So, we then grab the tag string:
docker-build: #pre-docker-builds
	git_branch=$$(git symbolic-ref --short -q HEAD) \
		|| git_branch=$$(git describe --tags)
	sudo docker build --network=host \
			--build-arg git_branch=$$git_branch \
			-t quay.io/hallamlab/saber:$$git_branch .

docker-run:
	git_branch=$$(git symbolic-ref --short -q HEAD) \
		|| git_branch=$$(git describe --tags)
	sudo docker run -it --network=host --rm \
		-v $(CURDIR):/input \
		-v $(CURDIR)/out:/output \
		quay.io/hallamlab/saber:$$git_branch bash 

singularity-local-build:
	git_branch=$$(git symbolic-ref --short -q HEAD) \
		|| git_branch=$$(git describe --tags)
	sudo /usr/local/bin/singularity build saber-$$git_branch.sif \
		docker-daemon://quay.io/hallamlab/saber:$$git_branch

