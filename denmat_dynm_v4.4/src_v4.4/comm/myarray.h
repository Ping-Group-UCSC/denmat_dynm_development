#pragma once
#include <scalar.h>
#include <constants.h>
#include <Random.h>

double** alloc_real_array(int n1, int n2, double val = 0.);
complex** alloc_array(int n1, int n2, complex val = c0);
complex*** alloc_array(int n1, int n2, int n3, complex val = c0);
complex**** alloc_array(int n1, int n2, int n3, int n4, complex val = c0);
void dealloc_real_array(double** arr);
void dealloc_array(complex** arr);

double** trunc_alloccopy_array(double** arr, int n1, int n2_start, int n2_end);
void trunc_copy_array(double** A, double **B, int n1, int n2_start, int n2_end);
// notice that in the following two subroutines, complex arrays are actually array matrices
// take arr[:, bStart:bEnd, bStart:bEnd] from arr[:, 0:n2, 0:n2]
complex** trunc_alloccopy_array(complex** arr, int n1, int n2, int bStart, int bEnd);
void trunc_copy_array(complex** A, complex **B, int n1, int n2, int bStart, int bEnd);

void zeros(double* arr, int n1);
void zeros(double** arr, int n1, int n2);
void zeros(complex* arr, int n1);
void zeros(complex** arr, int n1, int n2);
void zeros(complex*** arr, int n1, int n2, int n3);
void zeros(complex**** arr, int n1, int n2, int n3, int n4);

void random_array(complex* a, int n1);

void conj(complex* a, complex* c, int n1);

double maxval(double *arr, int n1);
double maxval(double **arr, int n1, int bStart, int bEnd);
double minval(double *arr, int n1);
double minval(double **arr, int n1, int bStart, int bEnd);

double dot(double *v1, double *v2, int n);