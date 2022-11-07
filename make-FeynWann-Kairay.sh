#!/bin/bash

module load myopenmpi-4.0.2_gcc-4.8.5

#CC=gcc CXX=g++ cmake \
# -D GSL_PATH=$GSL_DIR \
dir_cmake="/export/data/share/jxu/libraries/cmake-3.20.0-rc5/build/bin"
CC=mpicc CXX=mpic++ ${dir_cmake}/cmake \
 -D JDFTX_BUILD="../build" \
 -D JDFTX_SRC="../jdftx-1.7-202209/jdftx" \
 -D GSL_PATH="/export/data/share/jxu/libraries/gsl-2.6" \
 -D FFTW3_PATH="/export/data/share/jxu/libraries/fftw-3.3.8" \
 -D EnableScaLAPACK=yes \
 -D EnablePETSc=yes \
 -D PETSC_PATH="/export/data/share/jxu/libraries/petsc_myopenmp/build" \
 -D MPISafeWrite=yes \
 -D EnableProfiling=yes \
../FeynWann-master

make -j12
