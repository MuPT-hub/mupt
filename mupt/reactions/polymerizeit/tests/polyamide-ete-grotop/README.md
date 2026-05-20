# End-to-end test of PolymerizeIt! infrastructure generation and reactive simulations for MPD-TMC polyamide

This is the test case for and end-to-end test of PolymerizeIt! infrastructure generation, system setup, and reactive simulation from pre-written GROMACS inputs.

## Contents

- .gro and .top files for MPD and TMC monomers and the dimer
- .itp force field parameter files
- .mdp parameter files for energy minimization and equilibration
- Optional scripts for setting up initial configuration for reactive simulation

## How to run

- run `python make_pi.py`
    - This will generate the initial infrastructure and write auxiliary scripts used by PolymerizeIt!
    - It will also scan the .itp files for the monomers and dimers to identify the reactive atom indices
- run `cd polyamide` followed by `source preprocess.sh`
    - This will preprocess the input files, changing the atom labels of atoms participating in the reaction
- Generate an initial configuration of MPD and TMC monomers. Add the configuration file to `gro-files/iter0.gro` and the system topology file to `topology/iter0.top`
    - This can be done manually, or automatically by running `source init.sh`
- run `source run-code.sh` to run PolymerizeIt!

## Expected output

- `polyamide/`: directory containing all of the infrastructure, auxiliary scripts, etc. needed for PolymerizeIt!
- `polyamide/processed_inputs.json`: Preprocessed information identifying the updates to be made in a post-reaction scenario
- `polyamide/gro-files/iter10.gro`: Output gro file with updated atom names and residue names, and product atoms deleted
- `polyadime/topology/iter10.top`: Output topology file with with residue names of new polymer chain formed, number of previous chains updated, and details of which atom indices of monomers participated in the reaction.
- `polyamide/topology/paXXX_gmx.itp`: The new molecules formed get the names `paXXX` and topology files are generated for them.
- `polyamide/polymerizeit.log`: Log file