Multiscale Polymer Toolkit
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/MuPT-Hub/mupt/workflows/CI/badge.svg?branch=main)](https://github.com/MuPT-Hub/mupt/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/MuPT_Hub/mupt/branch/main/graph/badge.svg)](https://codecov.io/gh/MuPT_Hub/mupt/branch/main)


Library of core components and functionality for the Multiscale Polymer Toolkit (MuPT)

### Installation
#### Prerequisites
Installation of the Multiscale Polymer Toolkit (MuPT) makes use of package/environment management systems such as [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) (recommended) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html); be sure you have one of these installed on your machine.

#### Base install
To create a virtual environment with a local install of the MuPT, first clone the toolkit in a directory of your choice from the command line via:
```sh
git clone https://github.com/MuPT-hub/mupt
cd mupt
```

Then set up a fully-featured environment by running:
```sh
mamba env create -f devtools/conda-envs/release-env.yml
pip install .
mamba activate mupt-env
```
Or a much lighter (but non-MD capable) env for toolkit-only testing by running:
```sh
mamba env create -f devtools/conda-envs/light-env.yml
pip install .
mamba activate mupt-lite
```

#### Developer install
Those developing for the toolkit or otherwise interested in playing around with the source code may like to have a "live" editable installation on their machine, which mirrors changes made in the source to the installed version of the toolkit.

To create an environment with such an install, [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repo, then run the following commands in the directory of your choice:
```sh
git clone <link-to-your-fork>
cd mupt
mamba env create -f devtools/conda-envs/dev-env.yml -n mupt-dev
pip install -e . --config-settings editable_mode=strict
mamba activate mupt-dev
```

### Examples
See the accompanying [examples repository](https://github.com/MuPT-hub/mupt-examples) for tutorials on usage of the toolkit

### Copyright

Copyright (c) 2024, Timotej Bernat


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
