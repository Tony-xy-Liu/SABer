# SABer

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1a2954edef114b81a583bb23ffba2ace)](https://app.codacy.com/gh/hallamlab/SABer?utm_source=github.com&utm_medium=referral&utm_content=hallamlab/SABer&utm_campaign=Badge_Grade_Dashboard)

SAG Anchored Binner for recruiting metagenomic reads using single-cell amplified genomes as references

Check out the [wiki](https://github.com/hallamlab/SABer/wiki) for tutorials and more information on SABer!!

### Install SABer and Dependencies
Currently the easiest way to install SABer is to use a conda virtual environment.  
This will require the installation of [Anaconda](https://www.anaconda.com/distribution/).  
Once Anaconda is installed, you can follow the directions below to install all dependencies and SABer within a conda environment.
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
If you would like to use a docker or singularity container of SABer they are available:

[Docker](not_available_yet) (Not availavble yet...)

[Singularity](not_available_yet) (Not availavble yet...)

They can also be build from scratch using the following commands:

Docker:
```sh
make docker-build
```
Singularity:
```sh
make singularity-local-build
``` 
