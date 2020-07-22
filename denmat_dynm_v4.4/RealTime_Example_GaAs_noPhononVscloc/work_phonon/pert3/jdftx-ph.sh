#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --job-name=jdftx%j
#SBATCH --account=cfn37607
#SBATCH --mail-user=jxu153@ucsc.edu
#SBATCH --mail-type=FAIL

module load openmpi

MPICMD="srun -n $SLURM_NTASKS"
DIRJ="/sdcc/u/jxuucsc/jdftx_codes/jdftx/build"

prfx=phonon
for i in 3
do
   export ip="$i"
   export phononParams="iPerturbation $i"
   ${MPICMD} ${DIRJ}/phonon -i ${prfx}-scf.in > ${prfx}.${i}.out
   cp ${prfx}.${i}.out ${prfx}-scf.${i}.out
   ${MPICMD} ${DIRJ}/phonon -i ${prfx}.in > ${prfx}.${i}.out
done
