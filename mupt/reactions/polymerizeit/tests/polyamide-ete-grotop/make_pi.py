#************* MODIFY INPUTS HERE *************#

dirname="polyamide"
monomers = ['mpd', 'tmc']
monomer_filenames = ['mpd', 'tmc']
repeat_units = ['dim']
repeat_unit_filenames = ['dim']
molecules = monomers + repeat_units
filenames = monomer_filenames + repeat_unit_filenames
forcefield = ['forcefield.itp', 'nonbond.itp', 'bonded.itp']
emin_file = 'emin.mdp'
equil_file = 'equil.mdp'
n_iterations = 10

reactions = [['mpd','tmc','dim']]
reaction_template_A = [['n','h','h']]
reaction_template_B = [['c','o','cl']]
product_remove_atoms_A = [['h']]
product_remove_atoms_B = [['cl']]
reactive_groups = [[reaction_template_A[i], reaction_template_B[i], product_remove_atoms_A[i], product_remove_atoms_B[i]] for i in range(len(reactions))]
distance_cutoff = 0.35

input_name = 'input-pa.txt'

#************* END MODIFY INPUTS *************#


import os

from mupt.reactions.polymerizeit import write_pi as wpi


wpi.make_pi(dirname, molecules, filenames, forcefield, emin_file, equil_file, input_name, n_iterations)

wpi.write_pi_input_gro(monomers, repeat_units, monomer_filenames, repeat_unit_filenames, reactions, reactive_groups, distance_cutoff, f'{dirname}/{input_name}', root=False)
