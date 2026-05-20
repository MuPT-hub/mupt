#!/bin/bash
source /usr/local/gromacs/bin/GMXRC

eval "$(conda shell.bash hook)"
conda activate mupt

cp input.inp polyamide/init-files/
cp init.py polyamide/init-files/

cd polyamide
cp topology/forcefield.itp init-files/forcefield.itp
cp topology/nonbond.itp init-files/nonbond.itp
cp topology/bonded.itp init-files/bonded.itp
cp topology/mpd_gmx.itp init-files/mpd_gmx.itp
cp topology/tmc_gmx.itp init-files/tmc_gmx.itp
cd init-files

# generate input files
python init.py -i input.inp

# run energy minimization
gmx grompp -f emin.mdp -c polymer.gro -p topol.top -o emin.tpr
gmx mdrun -v -deffnm emin # -ntmpi 1 -ntomp 1

# if minimization not complete, exit
if [ ! -f emin.gro ]; then
    echo "Energy minimization did not complete successfully. Exiting."
    exit 1
fi

# run NVT equilibration
gmx grompp -f nvt.mdp -c emin.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt # -ntmpi 1 -ntomp 4

# if NVT not complete, exit
if [ ! -f nvt.gro ]; then
    echo "NVT equilibration did not complete successfully. Exiting."
    exit 1
fi

# run NPT equilibration
gmx grompp -f npt.mdp -c nvt.gro -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt # -ntmpi 1 -ntomp 4

# if NPT not complete, exit
if [ ! -f npt.gro ]; then
    echo "NPT equilibration did not complete successfully. Exiting."
    exit 1
fi

rm \#*

cp npt.gro ../gro-files/iter0.gro
cp topol.top ../topology/iter0.top

cd ../
