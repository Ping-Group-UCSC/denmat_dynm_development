#!/bin/bash
#SBATCH -p windfall
#SBATCH --account=windfall
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH -J jdftx

module load myopenmpi-4.0.2_gcc gsl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/data/groups/ping/jxu153/codes/jdftx/jdftx-202103/build"
DIRF="/data/groups/ping/jxu153/codes/jdftx/jdftx-202103/build-FeynWann"

#${MPICMD} ${DIRJ}/jdftx -i totalE.in > totalE.out
${MPICMD} ${DIRJ}/phonon -ni phonon.in > split.out
