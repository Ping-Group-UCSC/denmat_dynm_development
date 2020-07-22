#!/bin/bash
#SBATCH -p skx-normal
#SBATCH -N 8
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node=2
#SBATCH -J jdftx%j
#SBATCH -o job%j.out
#SBATCH -e job%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jxu153@ucsc.edu

module load gsl fftw3

MPICMD="ibrun"
DIRJ="/home1/06235/tg855346/jdftx_codes/jdftx-20181024/build"
DIRF="/home1/06235/tg855346/jdftx_codes/jdftx-20181024/build-FeynWann-20190528"

prfx=phonon
for i in 4
do
   export ip="$i"
   export phononParams="iPerturbation $i"
   ${MPICMD} ${DIRJ}/phonon -i ${prfx}-scf.in > ${prfx}.${i}.out
   cp ${prfx}.${i}.out ${prfx}-scf.${i}.out2
   #${MPICMD} ${DIRJ}/phonon -i ${prfx}.in > ${prfx}.${i}.out
done
