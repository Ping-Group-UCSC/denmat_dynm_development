#!/bin/bash
##SBATCH -p debug
#SBATCH -N 12
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=16
#SBATCH -J lindbladInit

#module purge
#module load gnu openmpi mkl gsl
module load myopenmpi-4.0.2_gcc-4.8.5

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/export/data/share/jxu/jdftx_codes/jdftx-202103/build"
DIRF="/export/data/share/jxu/jdftx_codes/jdftx-202103/build-FeynWann"

${MPICMD} ${DIRF}/lindbladInit_for-DMD-4.5.2 -i lindbladInit.in > lindbladInit.out
