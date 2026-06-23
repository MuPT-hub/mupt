from mupt.mudrivers.gromacs import (
    groio,
)
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GROMACS driver for MuPT.")
    parser.add_argument('-i','--inputs', type=str, required=True, help='Path to the input file containing parameters.')
    return parser.parse_args()

args = parse_args()

# read input parameters from file
inputs_dict = groio.read_inputs(args.inputs)

# use following line to create gro files
groio.create_gro('polymer', **inputs_dict)

# use following line to write top files
groio.write_top('topol', **inputs_dict)

# use following line to write mdp files
groio.write_mdp(**inputs_dict)
