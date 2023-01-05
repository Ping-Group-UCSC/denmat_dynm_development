#!/bin/bash
#SBATCH -p windfall
#SBATCH --account=windfall
#SBATCH -N 7
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=2
#SBATCH -J jdftx

module load myopenmpi-4.0.2_gcc gsl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/data/groups/ping/jxu153/codes/jdftx/jdftx-202103/build"
DIRF="/data/groups/ping/jxu153/codes/jdftx/jdftx-202103/build-FeynWann"

i=1
export phononParams="iPerturbation $i"
${MPICMD} ${DIRJ}/phonon -i phonon-scf.in > phonon-scf.${i}.out
${MPICMD} ${DIRJ}/phonon -i phonon.in > phonon.${i}.out
