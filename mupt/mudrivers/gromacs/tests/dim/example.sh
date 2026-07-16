# generate input files
python example.py -i inputs.inp

# run energy minimization
gmx grompp -f emin.mdp -c polymer.gro -p topol.top -o emin.tpr -maxwarn 1
gmx mdrun -v -deffnm emin # -ntmpi 1 -ntomp 1

# if minimization not complete, exit
if [ ! -f emin.gro ]; then
    echo "Energy minimization did not complete successfully. Exiting."
    exit 1
fi

rm \#*
