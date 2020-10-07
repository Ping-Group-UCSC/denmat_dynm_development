#!/bin/bash
#SBATCH -p skx-normal
#SBATCH -N 4
#SBATCH -t 02:40:00
#SBATCH --ntasks-per-node=48
#SBATCH -J jdftx%j
#SBATCH -o job%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jxu153@ucsc.edu

module load gsl fftw3

MPICMD="ibrun"
DIRJ="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build"
DIRF="/home1/06235/tg855346/jdftx_codes/jdftx-20190827/build-FeynWann-ePhEwind"

${MPICMD} ${DIRF}/electronPhononLinewidth -i ePhLw.in -G4 > ePhLw.out
${MPICMD} ${DIRF}/resistivity -i resistivity_elec.in -G4 > resistivity_elec.out
${MPICMD} ${DIRF}/resistivity -i resistivity_hole.in -G4 > resistivity_hole.out
