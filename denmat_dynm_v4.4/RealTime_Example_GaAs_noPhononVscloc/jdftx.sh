#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH -J jdftx

module load gsl fftw3

MPICMD="ibrun"
DIRJ="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build"
DIRF="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build-FeynWann-hist"

${MPICMD} ${DIRJ}/jdftx -i scf.in > scf.out
${MPICMD} ${DIRJ}/jdftx -i totalE.in > totalE.out
${MPICMD} ${DIRJ}/jdftx -i bandstruct.in > bandstruct.out
#${MPICMD} ${DIRJ}/phonon -ni phonon.in > phonon.out
