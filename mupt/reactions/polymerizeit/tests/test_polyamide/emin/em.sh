export OMP_NUM_THREADS=1

gmx grompp -f em.mdp -c ../gro-files/iterAAAA.gro -p ../topology/iterAAAA.top -po em-iterAAAA.mdp -o em-iterAAAA.tpr -maxwarn 1

gmx mdrun -s em-iterAAAA.tpr -v -deffnm em-iterAAAA -ntmpi 1 -ntomp 1

gmx trjconv -f em-iterAAAA.gro -s em-iterAAAA.tpr -o em-whole-iterAAAA.gro -pbc whole <<EOF
0
EOF

