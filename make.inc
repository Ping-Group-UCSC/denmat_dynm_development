GSL_DIR=/home/jxu153/work/libraries/gsl-2.6
IFLAGS=-I$(GSL_DIR)/include -I$(GSL_DIR)/include/gsl/gsl_odeiv2.h -I$(MKLROOT)/include
CC=mpiicpc -std=c++11 -O2 -g -traceback -mkl=sequential $(IFLAGS)
GSL_LIBS=-L$(GSL_DIR)/lib -lgsl -lgslcblas
LAPACK_LIBS=-L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LDLIBS=$(GSL_LIBS) $(LAPACK_LIBS)
