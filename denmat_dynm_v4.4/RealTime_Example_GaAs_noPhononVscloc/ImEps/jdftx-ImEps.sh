#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -N 4
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=12
#SBATCH -J ImEps

module load gsl fftw3

MPICMD="ibrun"
DIRJ="/home1/06235/tg855346/jdftx_codes/jdftx-20191025/build"
DIRF="/home1/06235/tg855346/jdftx_codes/jdftx-20191025/build-FeynWann-lindblad-mod"

${MPICMD} ${DIRF}/ImEps -i ImEpsLC.in > ImEpsLC.out
mv ImEpsDirect.dat ImEpsDirectLC.dat
${MPICMD} ${DIRF}/ImEps -i ImEpsRC.in > ImEpsRC.out
mv ImEpsDirect.dat ImEpsDirectRC.dat
