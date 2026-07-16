#!/bin/bash
source /usr/local/gromacs/bin/GMXRC

eval "$(conda shell.bash hook)"
conda activate mupt

rm -f sqlite_memdb.db

cp init-files/npt.gro gro-files/iter0.gro
cp init-files/topol.top topology/iter0.top

python polyamide.py  -inputs_file inputs-pa.txt \
                     -preprocess_database init.db \
                     -main_database sqlite_memdb.db

