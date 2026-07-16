# PolymerizeIt! reactions in GROMACS setup from MuPT representation

This subpackage contains functions necessary for setting up PolymerizeIt! reactions through the GROMACS MD engine from the MuPT reperesentation

## Contents

- `write_pi.py`: Contains functions for writing the input files for PolymerizeIt! and setting up the expected directory structure
    - `make_pi()`: Creates the expected directory structure, arranging .gro, .itp, .mdp, and other files and writting auxiliary scripts
    - `write_pi_input_gro()`: Writes the input file for PolymerizeIt! when using prewritten .gro and .itp files for the monomers and dimer
    - `write_pi_input_rep()`: Writes the input file for PolymerizeIt! when .gro and .itp files are written from SMILES strings
- `react_pi.py`: Contains functions for reacting monomers into the dimer from input SMILES strings and reaction template SMARTS strings
    - `react_monomers_to_dimer()`: Reacts monomers into the dimer and creates RDKit molecules and MuPT primitives for the monomers and dimer
    - `identify_reactive_sites()`: Identifies the indices in the monomers of the reacting atoms and corresponding indices in the dimer
    - `map_dimer_atoms_to_monomers()`: Idenifies which monomer atoms in the dimer came from
    - `openff_topol()`: Sets up an openff topology for atom typing
    - `openff_atom_typing()`: Performs atom typing of the monomers and dimer (usually takes a few minutes)
    - `combine_atom_types()`: Combines the parameters of common atom types between the molecules into a single .itp file
    - `write_itp()`: Writes the .itp file for the molecules using the names of the common atom types
- PolymerizeIt! protocol scripts: python scripts that provide a modular format for performing reactions in a specific manner
    - `gmx_simple.py`: The most basic flowchart - check for reaction using the specified reaction criteria for every cycle

## How to use

Examples of using the functions to perform MPD-TMC polyamide reactions from either 1) prewritten .gro and .itp files for the monomers and dimer or 2) SMILES strings of monomers and reaction template SMARTS strings are in `tests/polyamide-ete-grotop` and `tests/polyamide-ete-rep`, respectively.

