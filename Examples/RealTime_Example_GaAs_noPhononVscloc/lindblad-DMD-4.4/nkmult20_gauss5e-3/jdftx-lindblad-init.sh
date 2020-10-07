#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -N 4
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH -J lindbladInit

module load gsl fftw3

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/export/data/share/jxu/jdftx_codes/jdftx-202004/build"
DIRF="/export/data/share/jxu/jdftx_codes/jdftx-202004/build-FeynWann-mod"

${MPICMD} ${DIRF}/lindbladInit_for-DMD-4.4 -i lindbladInit.in > lindbladInit.out
