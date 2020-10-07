#!/bin/bash

module purge
module load gnu openmpi mkl gsl

CC=gcc CXX=g++ cmake \
 -D JDFTX_BUILD="../build" \
 -D JDFTX_SRC="../jdftx-master/jdftx" \
 -D GSL_PATH=$GSL_DIR \
 -D FFTW3_PATH="/export/data/share/jxu/libraries/fftw-3.3.8_gnu" \
../FeynWann-master-mod

make -j12
