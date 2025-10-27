Multiscale Polymer Toolkit
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/MuPT_Hub/mupt/workflows/CI/badge.svg)](https://github.com/MuPT_Hub/mupt/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/MuPT_Hub/mupt/branch/main/graph/badge.svg)](https://codecov.io/gh/MuPT_Hub/mupt/branch/main)


Drafting repository for the core functionality of the Multiscale Polymer Toolkit (MuPT)

### Installation
#### Prerequisites
Installation of the Multiscale Polymer Toolkit (MuPT) makes use of package/environment management systems such as [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) (recommended) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html); be sure you have one of these installed on your machine.

#### Base install
To create a virtual environment with a fully-featured local install of the MuPT, run the following command line instructions in the directory of choice on your machine:
```sh
git clone https://github.com/MuPT-hub/mupt
cd mupt
mamba env create -f devtools/conda-envs/release-env.yml -n mupt-env
pip install .
mamba activate mupt-env
```

#### Developer install
Those developing for the toolkit or otherwise interested in playing around with the source code may like to have a "live" editable installation on their machine, which mirrors changes made in the source to the installed version of the toolkit.

To create an environment with such an install, [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repo, then run the following commands in the directory of your choice:
```sh
git clone <link-to-your-fork>
cd mupt
mamba env create -f devtools/conda-envs/release-env.yml -n mupt-env
pip install -e . --config-settings editable_mode=strict
mamba activate mupt-env
```


### Copyright

Copyright (c) 2024, Timotej Bernat


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
