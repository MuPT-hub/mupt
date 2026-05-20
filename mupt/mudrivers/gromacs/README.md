# mupt.mudrivers.gromacs

A set of functions to convert from MuPT primitives to set up simulations using the GROMACS simulation engine.

-------------------------
### Requirements:

- mupt
- GROMACS (any version)

-------------------------
### Contents:

- `groio.py`
    - The main functions used to set up gromacs scripts
    - `create_gro`: create a gro file and add specified number of monomers
    - `write_mdp`: function to generate mdp files specified in the input files
    - `write_top`: function to generate the top files based on input specifications
    - `read_inputs`: reads the inputs and organizes it in a python dictionary

-----------------------

### Example:

- In the tests folder, execute `bash example.sh`
- First runs `example.py`
    - This uses an example input file `inputs.inp`
    - Writes `polymer.gro` configuration file
    - Writes `emin.mpd`, `nvt.mdp`, and `npt.mdp` parameter files
    - Writes `topol.top` topology file
- Then performs energy minimization, and NVT and NpT equilibration
