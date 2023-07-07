# SABer

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1a2954edef114b81a583bb23ffba2ace)](https://app.codacy.com/gh/hallamlab/SABer?utm_source=github.com&utm_medium=referral&utm_content=hallamlab/SABer&utm_campaign=Badge_Grade_Dashboard)

SAG Anchored Binner for recruiting metagenomic reads using single-cell amplified genomes as references

Check out the [wiki](https://github.com/hallamlab/SABer/wiki) for tutorials and more information on SABer!!

### Install SABer and Dependencies
Currently the easiest way to install SABer is to use a conda virtual environment.  
This will require the installation of conda. It can be [Anaconda](https://www.anaconda.com/download), [MambaForge](https://mamba.readthedocs.io/en/latest/installation.html), or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).\
Note: that SABer is written in Python 3, so the conda has to support Python 3.\
**Warning!!** SABer has been developed and tested on Linux, use OSX and Windows at your own risk!!

Once one of the "conda"s is installed, you can follow the directions below to install all dependencies and SABer within a conda environment.
```sh
git clone https://github.com/hallamlab/SABer.git
cd SABer
```
 Now use `make` to create the conda env, activate it, and install SABer via pip.
```sh
make install-saberenv
conda activate saber_cenv
make install-saber
```

### Test SABer Install
Here is a small [demo dataset](https://drive.google.com/file/d/1yUoPpoNRl6-CZHkRoUYDbikBJk4yC-3V/view?usp=sharing) to make sure your SABer install was successful.
Just download and follow along below to run SABer. (make sure you've activated the SABer conda env)
```sh
unzip demo.zip
cd demo
saber recruit -m k12.gold_assembly.fasta -l read_list.txt -o SABer_out -s SAG
```
The result of the above commands is a new directory named `SABer_out` that contains all the intermediate and final outputs for the SABer analysis. 

### Docker and Singularity containers
If you would like to use a [Docker](https://docs.docker.com/engine/install/) or [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) container of SABer they are available.\
In either case, they can be pulled from [Quay.IO](https://quay.io/repository/hallamlab/saber) with one of the following commands:
```sh
# For Docker (need sudo access)
sudo docker pull quay.io/hallamlab/saber
# Run above demo with Docker
sudo docker run -it --network=host --rm -v ./:/cwd quay.io/hallamlab/saber:latest saber recruit -m cwd/k12.gold_assembly.fasta -l cwd/docker_read_list.txt -o cwd/SABer_out -s cwd/SAG

#For Singularity
singularity pull docker://quay.io/hallamlab/saber
# Run the above demo with Singularity
singularity exec saber_latest.sif saber recruit -m k12.gold_assembly.fasta -l read_list.txt -o SABer_out -s SAG
```
Note: Make sure you are in the `demo` directory when running the above commands as the examples assume this.

They can also be build from scratch using the following commands:\
Docker (need sudo access):
```sh
make docker-build
```
Singularity (assumes you built the Docker locally first):
```sh
make singularity-local-build
``` 
