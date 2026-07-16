#source gromacs
#statement

cd equil
sed s/AAAA/$1/g equil.sh > temp-equil.sh
source ./temp-equil.sh
rm -f \#*
rm -f temp-equil.sh
cd ..
