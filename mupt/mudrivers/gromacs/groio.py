'''Import and export functions for performing GROMACS simulations'''

__author__ = 'Naomi Trampe'
__email__ = 'tramp037@umn.edu'

import subprocess

def write_mdp(**kwargs):
    """
    Writes parameters to a GROMACS .mdp file.

    Parameters:
    filename (str): The path to the output .mdp file.
    **kwargs: Key-value pairs representing the parameters to write.
    """

    if 'mdp_filename' not in kwargs:
        filenames = [f'mdp_{i}' for i in range(int(kwargs['nprocess'][0]))]
    else:
        filenames = kwargs['mdp_filename']
         
    n = int(kwargs['nprocess'][0])
    for i in range(n):
        with open(f"{filenames[i]}.mdp", 'w') as f:
            f.write("; integrator\n")
            f.write(f"{"integrator":<20}=   {kwargs['integrator'][i]}\n")
            f.write("\n; simulation time\n")
            f.write(f"{"nsteps":<20}=   {kwargs['nsteps'][i]}\n")
            if kwargs['emin'][i] == 'F':
                f.write(f"{"dt":<20}=   {kwargs.get('dt', [0.001 for _ in range(n)])[i]}\n")
            if kwargs['emin'][i] == 'T':
                f.write("\n; energy minimization parameters\n")
                f.write(f"{"emtol":<20}=   {kwargs.get('emtol', [10 for _ in range(n)])[i]}\n")
                f.write(f"{"emstep":<20}=   {kwargs.get('emstep', [0.01 for _ in range(n)])[i]}\n")
            f.write(f"\n; output control\n")
            output = [kwargs.get('nstxout', [0 for _ in range(n)])[i], 
                      kwargs.get('nstvout', [0 for _ in range(n)])[i], 
                      kwargs.get('nstfout', [0 for _ in range(n)])[i], 
                      kwargs.get('nstlog', [0 for _ in range(n)])[i], 
                      kwargs.get('nstenergy', [0 for _ in range(n)])[i], 
                      kwargs.get('nstxout-compressed', [0 for _ in range(n)])[i]]
            outnames = ["nstxout", "nstvout", "nstfout", "nstlog", "nstenergy", "nstxout-compressed"]
            for o in range(len(output)):
                if output[o] != "None":
                    f.write(f"{outnames[o]:<20}=   {output[o]}\n")
            f.write("\n; PBC and cutoffs\n")
            f.write(f"{"pbc":<20}=   {kwargs.get('pbc', ['xyz' for _ in range(n)])[i]}\n")
            f.write(f"{"rvdw":<20}=   {kwargs.get('rvdw', [1.0 for _ in range(n)])[i]}\n")
            f.write(f"{"rcoulomb":<20}=   {kwargs.get('rcoulomb', [1.0 for _ in range(n)])[i]}\n")
            f.write(f"{"vdwtype":<20}=   {kwargs.get('vdwtype', ['Cut-off' for _ in range(n)])[i]}\n")
            f.write(f"{"coulombtype":<20}=   {kwargs.get('coulombtype', ['PME' for _ in range(n)])[i]}\n")
            f.write(f"{"DispCorr":<20}=   {kwargs.get('DispCorr', 'EnerPres')}\n")
            if kwargs['emin'][i] == 'F':
                ensemble = kwargs.get('ensemble', ['NVE' for _ in range(n)])[i]
                if ensemble == 'NVT' or ensemble == 'NPT':
                    f.write("\n; temperature coupling\n")
                    f.write(f"{"tcoupl":<20}=   {kwargs.get('tcoupl', ['V-rescale' for _ in range(n)])[i]}\n")
                    f.write(f"{"tc-grps":<20}=   {kwargs.get('tc-grps', ['System' for _ in range(n)])[i]}\n")
                    f.write(f"{"tau_t":<20}=   {kwargs.get('tau_t', [0.1 for _ in range(n)])[i]}\n")
                    f.write(f"{"ref_t":<20}=   {kwargs.get('ref_t', [298.15 for _ in range(n)])[i]}\n")
                if ensemble == 'NPT':
                    f.write("\n; pressure coupling\n")
                    f.write(f"{"pcoupl":<20}=   {kwargs.get('pcoupl', ['Berendsen' for _ in range(n)])[i]}\n")
                    f.write(f"{"pcoupltype":<20}=   {kwargs.get('pcoupltype', ['isotropic' for _ in range(n)])[i]}\n")
                    f.write(f"{"tau_p":<20}=   {kwargs.get('tau_p', [1.0 for _ in range(n)])[i]}\n")
                    f.write(f"{"ref_p":<20}=   {kwargs.get('ref_p', [1.01325 for _ in range(n)])[i]}\n")
                    f.write(f"{"compressibility":<20}=   {kwargs.get('compressibility', [4.5e-5 for _ in range(n)])[i]}\n")
                f.write("\n; velocity generation\n")
                f.write(f"{"gen-vel":<20}=   {kwargs.get('gen-vel', ['yes' for _ in range(n)])[i]}\n")
                if kwargs.get('gen-vel', ['yes' for _ in range(n)])[i] == 'yes':
                    f.write(f"{"gen-temp":<20}=   {kwargs.get('ref_t', [298.15 for _ in range(n)])[i]}\n")
                    f.write(f"{"gen-seed":<20}=   {kwargs.get('gen-seed', [-1 for _ in range(n)])[i]}\n")

            if 'constraints' in kwargs and kwargs['constraints'][i] != 'None':
                f.write("\n; constraints\n")
                f.write(f"{"constraints":<20}=   {kwargs['constraints'][i]}\n")
                f.write(f"{"constraint_algorithm":<20}=   {kwargs.get('constraint_algorithm', ['lincs' for _ in range(n)])[i]}\n")
            
        print("Finished writing mdp file:", f"{filenames[i]}.mdp")


def write_top(filename, **kwargs):
    """
    Writes a GROMACS .top file based on provided parameters.

    Parameters:
    filename (str): The path to the output .top file.
    **kwargs: Key-value pairs representing the parameters to write.
    """

    with open(f"{filename}.top", 'w') as f:
        f.write("; GROMACS topology file\n\n")
        f.write(f'#include "{kwargs['ff'][0]}"\n\n')
        for i in range(len(kwargs.get('molecule_name',[]))):
            f.write(f'#include "{kwargs["molecule_topology"][i]}"\n')
            
        f.write("\n[ system ]\n")
        f.write(f"; Name\nMuPT\n\n")

        f.write("[ molecules ]\n")
        f.write(f"; {"Compound":<12}#mols\n")
        for name, n in zip(kwargs.get('molecule_name', []), kwargs.get('nmolecules', [])):
            f.write(f"{name:<14}{n}\n")
    
    print("Finished writing top file:", f"{filename}.top")


def create_gro(filename, **kwargs):
    """
    Creates a GROMACS .gro file based on provided parameters.

    Parameters:
    filename (str): The path to the output .gro file.
    **kwargs: Key-value pairs representing the parameters to use for creation.
    """

    box_dim = kwargs.get('box_dim', [1,1,1])
    molecule_name = kwargs.get('molecule_name', [])
    molecule_gro = kwargs.get('molecule_gro', [])
    nmolecules = kwargs.get('nmolecules', [])
    # Additional parameters can be added as needed

    print(f"Creating GRO file '{filename}' with box dimensions {box_dim}, "
          f"molecule names {molecule_name}, molecule GRO files {molecule_gro}, "
          f"and number of molecules {nmolecules}.")
    # Actual implementation to create the GRO file would go here

    # run the following bash command for each molecule type: gmx insert-molecules -ci ../init-emin/em-whole-dim.gro -box 20 20 30 -nmol 10 -o dim10.gro
    if len(molecule_name) > 0:
        for name, gro, n, i in zip(molecule_name, molecule_gro, nmolecules, range(len(molecule_name))):
            if i == 0:
                cmd = f"gmx insert-molecules -ci {gro} -box {box_dim[0]} {box_dim[1]} {box_dim[2]} -nmol {n} -o {filename}_{name}.gro"
                print(f"Running command: {cmd}")

                subprocess.run(cmd, shell=True, check=True)
            else:
                cmd = f"gmx insert-molecules -f {filename}_{molecule_name[i-1]}.gro -ci {gro} -box {box_dim[0]} {box_dim[1]} {box_dim[2]} -nmol {n} -o {filename}_{name}.gro"
                print(f"Running command: {cmd}")

                subprocess.run(cmd, shell=True, check=True)

        cmd = f"cp {filename}_{molecule_name[-1]}.gro {filename}.gro"
        subprocess.run(cmd, shell=True, check=True)
    elif len(molecule_name) == 0:
        with open(filename + '.gro', 'w') as f:
            f.write("\n")
            f.write("0\n")
            f.write(f"{box_dim[0]:.3f} {box_dim[1]:.3f} {box_dim[2]:.3f}\n")

def read_inputs(filename):
    """
    Reads input parameters from a mudrivers/gromacs input file.

    Parameters:
    filename (str): The path to the input file.

    Returns:
    dict: A dictionary containing the input parameters.
    """
    
    inputs = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#') and not line.startswith(';'):
                key, value = line.split('=')
                inputs[key.strip()] = value.strip().split(',')
    return inputs

def read_gro(filename):
    """
    Reads a GROMACS .gro file and returns the coordinates and box dimensions.

    Parameters:
    filename (str): The path to the .gro file.

    Returns:
    """

def write_gro(filename, box=None):
    """
    Writes coordinates and box dimensions to a GROMACS .gro file.
    
    Parameters:
    filename (str): The path to the output .gro file.
    box (list or None): The box dimensions to write, if any.
    """