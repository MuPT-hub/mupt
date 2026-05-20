#************* MODIFY INPUTS HERE *************#

# Define SMILES strings for the molecules
mon_A_smi = 'c1ccc(N)cc1N'
mon_B_smi = 'c1c(C(=O)Cl)cc(C(=O)Cl)cc1C(=O)Cl'
mon_names = ['MPD', 'TMC']
dim_name = 'DIM'

# Define the pieces of the reaction template
react_template_A = '[N:1]'
react_idx_A = 0
react_template_B = '[C:2](=[O:3])Cl'
react_idx_B = 0
prod_template = '[N:1][C:2]=[O:3]'
prod_idx_A = 0
prod_idx_B = 1

# PolymerizeIt! inputs
dirname="polyamide"
emin_file = 'emin.mdp'
equil_file = 'equil.mdp'
n_iterations = 10
distance_cutoff = 0.35


#************* END MODIFY INPUTS *************#
from mupt.reactions.polymerizeit import react_pi
from mupt.reactions.polymerizeit import write_pi

# React the monomers to form the dimer and create primitives and RDKit molecules
mon_A_prim, mon_A, mon_B_prim, mon_B, dim_prim, dim, dim_smi = react_pi.react_monomers_to_dimer(mon_A_smi, mon_B_smi, react_template_A, react_template_B, prod_template)

# find atom indices corresponding to reactive sites in monomers and dimer
mon_A_react, mon_B_react, mon_A_prod, mon_B_prod = react_pi.identify_reactive_sites(mon_A, mon_B, dim, 
                                                                             react_template_A, react_idx_A, 
                                                                             react_template_B, react_idx_B, 
                                                                             prod_template, prod_idx_A, prod_idx_B)

# match the atom indices with monomer A or B
mon_A_indices, mon_B_indices = react_pi.map_dimer_atoms_to_monomers(dim_prim, mon_A_prod, mon_B_prod)

# assign atom types
# this will take a few minutes
react_pi.openff_atom_typing(mon_names, mon_A_smi, mon_B_smi, dim_smi)

# combine atom types from each molecule
atom_types_dict = react_pi.combine_atom_types(mon_names)

# write molecule and forcefield .itp files
react_pi.write_itp(mon_names, atom_types_dict)

monomers = [mon_names[0].lower(), mon_names[1].lower()]
monomer_filenames = [mon_names[0].lower(), mon_names[1].lower()]
repeat_units = [dim_name.lower()]
repeat_unit_filenames = [dim_name.lower()]
molecules = monomers + repeat_units
filenames = monomer_filenames + repeat_unit_filenames
ff_files = ['forcefield.itp', 'nonbond.itp', 'bonded.itp']
input_name = 'inputs-pa.txt'
reactions = [[mon_names[0].lower(), mon_names[1].lower(), dim_name.lower()]]

write_pi.make_pi(dirname, molecules, filenames, ff_files, emin_file, equil_file, input_name, n_iterations)

write_pi.write_pi_input_repr(monomers, repeat_units, monomer_filenames, repeat_unit_filenames, reactions, [[mon_A_react, mon_B_react, mon_A_prod, mon_B_prod]], [[mon_A_indices, mon_B_indices]], distance_cutoff, f'{dirname}/{input_name}', root=False)