#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -N 4
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=48
#SBATCH -J dm

module load gsl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRDM="/work/06235/tg855346/stampede2/denmat-codes/eph/bin"

$MPICMD $DIRDM/denmat_dynm_v4.4 > out
