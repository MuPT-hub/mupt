# End-to-end test of PolymerizeIt! infrastructure generation and reactive simulations for MPD-TMC polyamide

This is the test case for and end-to-end test of PolymerizeIt! infrastructure generation, system setup, and reactive simulation from pre-written SMILES strings and reaction template SMARTS strings.

## Contents

- `make_pi.py` a prewritten python script containing SMILES strings and reaction template SMARTS strings for MPD-TMC polyamide reaction
- .mdp parameter files for energy minimization and equilibration
- Optional scripts for setting up initial configuration for reactive simulation

## How to run

- run `python make_pi.py`
    - This script will predict the dimer formed from the reaction through RDKit functions and write out .gro and .top files using openff atom typing
    - Aditionally, it will generate the initial infrastructure and write auxiliary scripts used by PolymerizeIt!
    - Finally, it will write the inputs file for running PolymerizeIt! with the .gro and .top files generated in the first step
- run `cd polyamide` followed by `source preprocess.sh`
    - This will preprocess the input files, changing the atom labels of atoms participating in the reaction
- Generate an initial configuration of MPD and TMC monomers. Add the configuration file to `gro-files/iter0.gro` and the system topology file to `topology/iter0.top`
    - This can be done manually, or automatically by running `source init.sh`
- run `source run-code.sh` to run PolymerizeIt!

## Expected output

- `.gro`, `.top`, and `.itp` files for the monomers and dimer
- `forcefield.itp`, `nonbond.itp`, and `bonded.itp` containing the force field parameters for the system of monomers and reaction products
- `polyamide/`: directory containing all of the infrastructure, auxiliary scripts, etc. needed for PolymerizeIt!
- `polyamide/processed_inputs.json`: Preprocessed information identifying the updates to be made in a post-reaction scenario
- `polyamide/gro-files/iter10.gro`: Output gro file with updated atom names and residue names, and product atoms deleted
- `polyadime/topology/iter10.top`: Output topology file with with residue names of new polymer chain formed, number of previous chains updated, and details of which atom indices of monomers participated in the reaction.
- `polyamide/topology/paXXX_gmx.itp`: The new molecules formed get the names `paXXX` and topology files are generated for them.
- `polyamide/polymerizeit.log`: Log file
