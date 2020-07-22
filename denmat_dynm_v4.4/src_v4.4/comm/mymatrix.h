#pragma once
#include <scalar.h>
#include <string>
#include <constants.h>
#include <vector3.h>
#include <gsl/gsl_cblas.h>
using namespace std;

//Lapack forward declarations
extern "C"
{
	void zheevr_(char* JOBZ, char* RANGE, char* UPLO, int * N, complex* A, int * LDA,
		double* VL, double* VU, int* IL, int* IU, double* ABSTOL, int* M,
		double* W, complex* Z, int* LDZ, int* ISUPPZ, complex* WORK, int* LWORK,
		double* RWORK, int* LRWORK, int* IWORK, int* LIWORK, int* INFO);
	void zgeev_(char* JOBVL, char* JOBVR, int* N, complex* A, int* LDA,
		complex* W, complex* VL, int* LDVL, complex* VR, int* LDVR,
		complex* WORK, int* LWORK, double* RWORK, int* INFO);
	void zgesdd_(char* JOBZ, int* M, int* N, complex* A, int* LDA,
		double* S, complex* U, int* LDU, complex* VT, int* LDVT,
		complex* WORK, int* LWORK, double* RWORK, int* IWORK, int* INFO);
	void zgesvd_(char* JOBU, char* JOBVT, int* M, int* N, complex* A, int* LDA,
		double* S, complex* U, int* LDU, complex* VT, int* LDVT,
		complex* WORK, int* LWORK, double* RWORK, int* INFO);
	void zgetrf_(int* M, int* N, complex* A, int* LDA, int* IPIV, int* INFO);
	void zgetri_(int* N, complex* A, int* LDA, int* IPIV, complex* WORK, int* LWORK, int* INFO);
	void zposv_(char* UPLO, int* N, int* NRHS, complex* A, int* LDA, complex* B, int* LDB, int* INFO);
}

void diagonalize(complex *H, int n, double *eig, complex *v);

void zhemm_interface(complex *C, bool left, complex *A, complex *B, int n, complex alpha = c1, complex beta = c0);

void zgemm_interface(complex *C, complex *A, complex *B, int n, complex alpha = c1, complex beta = c0, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans);
void zgemm_interface(complex *C, complex *A, complex *B, int m, int n, int k, complex alpha = c1, complex beta = c0, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans);

void aij_bji(double *C, complex *A, complex *B, int n, double alpha = 1, double beta = 0);

void mat_diag_mult(complex *C, complex *A, double *B, int n);
void mat_diag_mult(complex *C, double *A, complex *B, int n);
void vec3_dot_vec3array(complex *vm, vector3<double> v, complex **m, int n);
void vec3_dot_vec3array(complex *vm, vector3<complex> v, complex **m, int n); // m[3][]

// b[0:(i1-i0), 0:(j1-j0)] = a[i0:i1, j0:j1] of a[:, 0:n]
void trunc_copy_mat(complex *b, complex *a, int n, int i0, int i1, int j0, int j1);
// b[i0:i1, j0:j1] (of b[:, 0:n]) = a[0:(i1-i0), 0:(j1-j0)]
void set_mat(complex *b, complex *a, int n, int i0, int i1, int j0, int j1);
void hermite(complex *m, complex *h, int n);
void hermite(complex *a, complex *h, int m, int n);