# Test case for MPD-TMC polyamide polymer

This is the test cas for 10 iterations of PolymerizeIt! performing the MPD-TMC polymerization reaction. Base monomer and repeat unit files are present in the init-files folder
This is the test case for and end-to-end test in an iteration of normal routine. The monomer and repeat unit files are generated using AmberTools and Parmed, and those scripts are present in the `init-files` folder. 

## How to run

- Activate mupt environment as per the instructions in the main README.md
- Ensure that GROMACS is loaded in the current environment
- To preprocess the inputs, enter `source preprocess.sh`
- To create the initial box, enter `source setup.sh`
- To run PolymerizeIt!, enter `source run-code.sh`

## Expected output

- `processed_inputs.json`: Preprocessed information identifying the updates to be made in a post-reaction scenario
- `gro-files/iter1.gro`: Output gro file with updated atom names and residue names, and product atoms deleted
- `topology/iter1.top`: Output topology file with with residue names of new polymer chain formed, number of previous chains updated, and details of which atom indices of monomers participated in the reaction.
- `topology/paaab_gmx.itp`: For the configuration in `iter0.gro`, one one reaction was possible. The new molecule formed gets the name `paaab` and the topology file is generated for it.
- `polymerizeit.log`: Log file