#!/bin/bash
##SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=16
#SBATCH -J jdftx

module use /home/jxu153/modulefiles
module load myopenmpi-4.0.2_gcc-4.8.5
#module purge
#module load gnu openmpi mkl gsl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/export/data/share/jxu/jdftx_codes/jdftx-feynwann-stable-202109/build"
DIRF="/export/data/share/jxu/jdftx_codes/jdftx-feynwann-stable-202109/build-FeynWann"

${MPICMD} ${DIRF}/lindbladLinear -i lindbladLinear.in -G8 > lindbladLinear.out
