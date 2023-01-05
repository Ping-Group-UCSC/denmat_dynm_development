#!/bin/bash
#SBATCH -p windfall
#SBATCH --account=windfall
#SBATCH -N 6
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=40
#SBATCH -J jdftx

module load myopenmpi-4.0.2_gcc gsl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/data/groups/ping/jxu153/codes/jdftx/jdftx-202103/build"
DIRF="/data/groups/ping/jxu153/codes/jdftx/jdftx-202103/build-FeynWann"

prfx=wannier
${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
exit
python rand_wann-centers.py
cp wannier.in0 wannier.in
cat rand_wann-centers.dat >> wannier.in
rm rand_wann-centers.dat
${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
