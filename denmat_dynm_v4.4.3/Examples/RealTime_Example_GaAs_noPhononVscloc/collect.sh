#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=4
#SBATCH -J jdftx%j
#SBATCH -o job%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jxu153@ucsc.edu

module load gsl fftw3

MPICMD="ibrun"
DIRJ="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build"
DIRF="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build-FeynWann-hist"

export phononParams="collectPerturbations"
${MPICMD} ${DIRJ}/phonon -i phonon.in > phonon.out
