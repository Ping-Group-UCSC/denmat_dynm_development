#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -N 4
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=48
#SBATCH -J jdftx%j
#SBATCH -o job%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jxu153@ucsc.edu

module load gsl fftw3

MPICMD="ibrun"
DIRJ="/home1/06235/tg855346/jdftx_codes/jdftx-20191025/build"
DIRF="/home1/06235/tg855346/jdftx_codes/jdftx-20191025/build-FeynWann-lindblad-mod"

prfx=wannier
${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
<<comm
if [ ! -f wannier.mlwfU ]; then
  isnew=1
else
  if grep -q "Probably at roundoff error limit" wannier.out; then
    isnew=1
  else
    isnew=0
    sed -i "s/loadRotations no/loadRotations yes/g" wannier.in
    ${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
  fi
fi
failed=1
ntry=0
while { [ $isnew == 1 ] && [ $failed == 1 ] ;} && [ $ntry -lt 2 ]
do
  ntry=$[ ntry + 1 ]
  python rand_wann-centers.py
  cp wannier.in0 wannier.in
  cat rand_wann-centers.dat >> wannier.in
  rm rand_wann-centers.dat
  ${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
  if grep -q "Probably at roundoff error limit" wannier.out; then
    failed=1
  else
    if [ ! -f wannier.mlwfU ]; then
      failed=1
      cp wannier.out wannier_failed.out
    else
      failed=0
    fi
  fi
done
comm
