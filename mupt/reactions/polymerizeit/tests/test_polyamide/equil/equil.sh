export OMP_NUM_THREADS=1

gmx grompp -f nvt-equil.mdp -c ../emin/em-whole-iterAAAA.gro -p ../topology/iterAAAA.top -po nvt-equil-iterAAAA.mdp -o nvt-equil-iterAAAA.tpr -maxwarn 1

gmx mdrun -s nvt-equil-iterAAAA.tpr -v -deffnm nvt-equil-iterAAAA  -ntomp 1 -ntmpi 1 -pin on

gmx trjconv -f nvt-equil-iterAAAA.xtc -s nvt-equil-iterAAAA.tpr -o nvt-equil-whole-iterAAAA.xtc -pbc whole <<EOF
0
EOF

gmx trjconv -f nvt-equil-iterAAAA.gro -s nvt-equil-iterAAAA.tpr -o nvt-equil-whole-iterAAAA.gro -pbc whole <<EOF
0
EOF
