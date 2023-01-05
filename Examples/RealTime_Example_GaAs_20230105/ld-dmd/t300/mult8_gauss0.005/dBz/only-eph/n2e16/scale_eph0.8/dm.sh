#!/bin/bash
##SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=16
#SBATCH -J dm

MPICMD="mpirun -np $SLURM_NTASKS"
DIRDM="/export/data/share/jxu/denmat-codes/bin"

$MPICMD $DIRDM/denmat_dynm_v4.5.6 > out
