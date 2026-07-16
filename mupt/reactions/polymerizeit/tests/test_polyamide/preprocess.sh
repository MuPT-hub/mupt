#!/bin/bash
source /usr/local/gromacs/bin/GMXRC

eval "$(conda shell.bash hook)"
conda activate mupt

rm -f sqlite_memdb.db
rm -f processed_inputs.json
rm -f init.db

cp ../../../../mudrivers/gromacs/tests/mpd-tmc/mpd.gro gro-files/
cp ../../../../mudrivers/gromacs/tests/mpd-tmc/tmc.gro gro-files/
cp ../../../../mudrivers/gromacs/tests/dim/dim.gro gro-files/

cp ../../../../mudrivers/gromacs/tests/mpd-tmc/mpd.itp gro-files/mpd.top
cp ../../../../mudrivers/gromacs/tests/mpd-tmc/tmc.itp gro-files/tmc.top
cp ../../../../mudrivers/gromacs/tests/dim/dim.itp gro-files/dim.top

cp ../../../../mudrivers/gromacs/tests/dim/forcefield.itp topology/forcefield.itp
cp ../../../../mudrivers/gromacs/tests/dim/nonbond.itp topology/nonbond.itp
cp ../../../../mudrivers/gromacs/tests/dim/bonded.itp topology/bonded.itp

python write_input.py

python process.py    -inputs_file inputs-pa.txt \
                     -preprocess_database init.db \
                     -main_database sqlite_memdb.db

