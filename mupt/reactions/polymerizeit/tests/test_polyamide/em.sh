#source gromacs
#statement

cd emin
sed s/AAAA/$1/g em.sh > temp-em.sh
source ./temp-em.sh
rm -f \#*
rm -f temp-em.sh
cd ..
