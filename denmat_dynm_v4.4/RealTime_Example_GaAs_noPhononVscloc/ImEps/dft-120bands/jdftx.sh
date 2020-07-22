#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -N 4
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH -J jdftx%j
#SBATCH -o job%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jxu153@ucsc.edu

module load gsl fftw3

MPICMD="ibrun"
DIRJ="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build"
DIRF="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build-FeynWann-hist"

${MPICMD} ${DIRJ}/jdftx -i scf.in > scf.out
${MPICMD} ${DIRJ}/jdftx -i totalE.in > totalE.out
${MPICMD} ${DIRJ}/jdftx -i bandstruct.in > bandstruct.out
#${MPICMD} ${DIRJ}/phonon -ni phonon.in > phonon.out
