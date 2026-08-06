from datetime import datetime #only for metadata['date'] on line 48
from pathlib import Path #for definfinf paths
from ruamel.yaml import YAML 
from ruamel.yaml.comments import CommentedSeq

from rdkit.Chem import Descriptors

# PolymerizeIt init asks for a bunch of things - project name, monomers, repeat units, and finally
# which "protocol starter" to use. That last question is the last three arguments here.
# The menu is: option 1 is "custom (build from scratch)", which is also the default, and options 2+
# are its reference protocols (worked examples like polyamide_kolev_03242014 that ship with it).
# We always want custom, because fill_config writes into the inputs.yaml that gets generated - a
# reference protocol would hand us someone else's already-filled config for different chemistry.
# Picking custom means passing None for all three. They have no defaults, so we must pass them.
CUSTOM_PROTOCOL = (None, None, None)


def _smiles_sources(molecules):
    '''Describe molecules to PolymerizeIt! as SMILES sources.

    PolymerizeIt! records where each molecule came from as a {source, value} pair, since it also
    accepts file paths and repository lookups. mupt has already resolved everything to SMILES.
    '''
    return [{'name': molecule['name'], 'source': 'smiles', 'value': molecule['smi']}
            for molecule in molecules]


def generate_config_file(inputs, system):
    '''Scaffold a PolymerizeIt! project directory for one system, and record it on `inputs`.

    Calls PolymerizeIt!'s project builder directly rather than driving its interactive CLI. The
    import is deliberately function-local: PolymerizeIt! is an external tool, not a mupt dependency,
    so this module must stay importable without it.
    '''
    from polymerizeit.commands.init import create_project_directory

    if 'dirname' not in inputs:
        print("No directory name provided in inputs. Using default directory name 'polymer'.")
        inputs['dirname'] = 'polymer'

    if 'protocol' not in inputs['reaction_engine']['inputs']:
        print("No protocol provided in reaction engine inputs. Using default protocol.")
        inputs['reaction_engine']['inputs']['protocol'] = 'default'
    if inputs['reaction_engine']['inputs']['protocol'] != 'default':
        print("Other protocols currently not supported through this interface. Using default protocol.")

    # PolymerizeIt! names the directory <polymer>_<author>_<date>; mupt supplies no author, so its
    # default applies. The path it returns is authoritative -- it deduplicates against existing
    # directories, so the name cannot be reliably reconstructed here.
    metadata = {
        'polymer_name': inputs['dirname'],
        'author_name': 'unknown',
        'date': datetime.today().strftime('%Y-%m-%d'),
    }
    project_path = Path(create_project_directory(
        metadata,
        _smiles_sources(inputs['monomers']),
        _smiles_sources(inputs['repeat_units']),
        *CUSTOM_PROTOCOL,
    ))

    # give the project its system-specific name, archiving any project left by a previous run
    directory = Path(f"{inputs['dirname']}_{system['name']}")
    if directory.exists():
        archived = rename_no_overwrite(directory, f"{directory}_archive")
        print(f"Directory {directory} already exists. Renaming it to {archived}.")

    project_path.rename(directory)

    inputs['directory'] = str(directory)
    return

def fill_config(inputs, system):
    
    print("\n\n\nUpdating config file with system parameters and reaction information...\n")
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(f"{inputs['dirname']}_{system['name']}/inputs.yaml") as f:
        data = yaml.load(f)



    # add the reactions to the config file with determined indices
    data['preprocessing']['reference_reactions'] = []
    for i, reaction in enumerate(inputs['reactions']):
        data['preprocessing']['reference_reactions'].append({'name': reaction})
        reactant_list = []
        for j, reactant in enumerate(inputs['reactions'][reaction]['reactants']):
            reactant_list.append({'monomer': f"{chr(65+j)}", 'atom': inputs['reactions'][reaction]['reactant_indices'][j]})
        data['preprocessing']['reference_reactions'][i]['reactants'] = reactant_list
        
        product_list = []
        for k, product in enumerate(inputs['reactions'][reaction]['product_indices']):
            product_list.append({'monomer': i+1, 'atom': inputs['reactions'][reaction]['product_indices'][k]})
        data['preprocessing']['reference_reactions'][i]['products'] = product_list

    data.yaml_set_comment_before_after_key('atom_name_same_as_atom_type', before='\n# ===== PREPROCESSING OPTIONS =====')



    # determine number of molecules in the initial system
    monomer_data = {}
    if 'n_monomers' in system:
        # calculate the number of atoms and mass of each monomer
        for monomer in system['n_monomers']:
            for molecule in inputs['molecules']:
                if molecule['name'] == monomer:
                    monomer_data[monomer]['n_atoms'] = len(molecule['prim'].topology.nodes())
                    monomer_data[monomer]['mass'] = round(Descriptors.MolWt(molecule['mol']), 3)
            monomer_data[monomer] = {'n_molecules': system['n_monomers'][monomer]}
    elif 'monomer_ratios' in system:
        # calculate the number of atoms and mass of each monomer
        for monomer in system['monomer_ratios']:
            monomer_data[monomer] = {'ratio': system['monomer_ratios'][monomer]}
            for molecule in inputs['molecules']:
                if molecule['name'] == monomer:
                    monomer_data[monomer]['n_atoms'] = len(molecule['prim'].topology.nodes())
                    monomer_data[monomer]['mass'] = round(Descriptors.MolWt(molecule['mol']), 3)
        if 'total_monomers' in system:
            print(f"Calculating number of each monomer from {system['total_monomers']} and monomer ratios.")
            # TODO
        elif 'total_atoms' in system:
            print(f"Calculating number of each monomer from {system['total_atoms']} and monomer ratios.")
            for monomer in monomer_data:
                monomer_data[monomer]['atom_ratio'] = monomer_data[monomer]['ratio']*monomer_data[monomer]['n_atoms']
            
            atom_ratio_sum = sum([monomer_data[monomer]['atom_ratio'] for monomer in monomer_data])

            for monomer in monomer_data:
                monomer_data[monomer]['n_molecules'] = int(monomer_data[monomer]['ratio']*system['total_atoms']/atom_ratio_sum)
        else:
            if 'box_size' in system:
                if 'density' in system:
                    print(f"No total atoms or total monomers provided in system parameters. Determining total from box size and density.")
                    # TODO
                else:
                    print(f"No total atoms or total monomers provided in system parameters. Determining total from box size and 1 g/cc density.")
                    # TODO
            else:
                print(f"No total atoms or total monomers provided in system parameters. Default total number of monomers is used {100*len(monomer_data.keys)}.")
                # TODO
    else:
        # calculate the number of atoms and mass of each monomer
        monomer_data = {}
        for monomer in inputs['monomers']:
            monomer_data[monomer] = {'ratio': 1.0}
            for molecule in inputs['molecules']:
                if molecule['name'] == monomer:
                    monomer_data[monomer]['n_atoms'] = len(molecule['prim'].topology.nodes())
                    monomer_data[monomer]['mass'] = Descriptors.MolWt(molecule['mol'])
        print("No monomer ratios provided in system parameters. Assuming equal ratios of each monomer.")
        if 'total_monomers' in system:
            print(f"Calculating number of each monomer from {system['total_atoms']} and equal monomer ratios.")
            # TODO
        elif 'total_atoms' in system:
            print(f"Calculating number of each monomer from {system['total_monomers']} and equal monomer ratios.")
            # TODO
        else:
            if 'box_size' in system:
                if 'density' in system:
                    print(f"No total atoms or total monomers provided in system parameters. Determining total from box size and density.")
                    # TODO
                else:
                    print(f"No total atoms or total monomers provided in system parameters. Determining total from box size and 1 g/cc density.")
                    # TODO
            else:
                print("No total atoms or total monomers provided in system parameters. Default of 100 of each is used.")
                # TODO

    for monomer in monomer_data:
        print(f"Number of molecule {monomer}: {monomer_data[monomer]['n_molecules']}")
        data['init_system']['n_molecules'][monomer] = monomer_data[monomer]['n_molecules']



    # determine box size
    if 'box_size' in system:
        # data['init_system']['box_dimensions'] = system['box_size']
        dims = CommentedSeq(system['box_size'])
    else:
        print("Box size not provided in system parameters. Calculating box size from density and number of monomers.")
        if 'density' not in system:
            print("Density not provided. Using default density (1 g/cc).")
            system['density'] = 1.0
        box_mass = 0
        for monomer in monomer_data:
            box_mass += monomer_data[monomer]['n_molecules']*monomer_data[monomer]['n_molecules'] / 6.022e23
        box_length = round((box_mass / system['density'] * 1e21) ** (1/3), 5)
        dims = CommentedSeq([box_length, box_length, box_length])
        # data['init_system']['box_dimensions'] = [box_length, box_length, box_length]
    
    dims.fa.set_flow_style()
    data['init_system']['box_dimensions'] = dims
    print(f"Box dimensions: {data['init_system']['box_dimensions']}")
        
        

    # update distance cutoff, if provided
    if 'distance_cutoff' in inputs['reaction_engine']['inputs']:
        data['reactions']['distance_cutoff'] = inputs['reaction_engine']['inputs']['distance_cutoff']
        print(f"Distance cutoff: {data['reactions']['distance_cutoff']}")
    else:
        print("No distance cutoff provided in reaction engine inputs. Using default distance cutoff.")



    # edit reaction criteria
    data['reactions']['detection_criteria'] = []
    if 'reaction_criteria' not in inputs['reaction_engine']['inputs']:
        print("No reaction criteria provided in reaction engine inputs. Using default reaction criteria.")
        inputs['reaction_engine']['inputs']['reaction_criteria'] = ['DistanceCutoff']
    for criterion in inputs['reaction_engine']['inputs']['reaction_criteria']:
        data['reactions']['detection_criteria'].append({'name': criterion})
    print(f"Reaction criteria: {inputs['reaction_engine']['inputs']['reaction_criteria']}")
    data['reactions'].yaml_set_comment_before_after_key('post_reaction_updates', before='\n')


    # update temperature
    if 'temperature' in system:
        data['equilibration']['nvt']['external_software_files']['nvt_mdp']['file_variables']['ref_t'] = system['temperature']
    else:
        print("No temperature provided in system parameters. Using default temperature.")
    print(f"Temperature: {data['equilibration']['nvt']['external_software_files']['nvt_mdp']['file_variables']['ref_t']}")

    # dump the updated config file
    with open(f"{inputs['dirname']}_{system['name']}/inputs.yaml", 'w') as f:
        yaml.dump(data, f)

    print(f"\n\nConfig file updated and saved to {inputs['dirname']}_{system['name']}/inputs.yaml\n")
    return

def rename_no_overwrite(src: str | Path, dst: str | Path) -> Path:
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(src)

    parent = dst.parent
    stem = dst.name

    candidate = parent / stem
    i = 1
    while candidate.exists():
        candidate = parent / f"{stem}_{i}"
        i += 1

    src.rename(candidate)
    return candidate