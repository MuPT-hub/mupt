Multiscale Polymer Toolkit
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/MuPT_Hub/mupt/workflows/CI/badge.svg)](https://github.com/MuPT_Hub/mupt/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/MuPT_Hub/mupt/branch/main/graph/badge.svg)](https://codecov.io/gh/MuPT_Hub/mupt/branch/main)


Drafting repository for the core functionality of the Multiscale Polymer Toolkit (MuPT)

### Installation
To create a virtual fully-featured environment with the Multiscale Polymer Toolkit, run the following commands from the command line in the desired directory on a machine with a Python installation:
```sh
git clone https://github.com/MuPT-hub/mupt
cd mupt
mamba env create -f devtools/conda-envs/release-env.yml
pip install .
```

For developers, can perform and editable install and mirror `mupt` to the current environment (i.e. allow changes to source to take effect in active environment) by replacing the final `pip` command with:
```sh
pip install -e . --config-settings editable_mode=strict
```


### Copyright

Copyright (c) 2024, Timotej Bernat


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
