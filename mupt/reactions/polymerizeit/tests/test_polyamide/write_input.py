from mupt.mutation.polymerizeit import write_pi

monomers = ['mpd', 'tmc']
monomer_filenames = ['gro-files/mpd', 'gro-files/tmc']
repeat_units = ['dim']
repeat_unit_filenames = ['gro-files/dim']
reactions = [['mpd','tmc','dim']]
reaction_template_A = ['n','h','h']
reaction_template_B = ['c','o','cl']
product_remove_atoms_A = ['h']
product_remove_atoms_B = ['cl']
reactive_groups = [[reaction_template_A, reaction_template_B, product_remove_atoms_A, product_remove_atoms_B]]
distance_cutoff = 0.35
outname = 'inputs-pa.txt'

write_pi.write_pi_input_gro(monomers, repeat_units, monomer_filenames, repeat_unit_filenames, reactions, reactive_groups, distance_cutoff, outname)