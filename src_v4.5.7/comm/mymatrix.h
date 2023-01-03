#pragma once
#include <scalar.h>
#include <string>
#include <constants.h>
#include <vector3.h>
#include <gsl/gsl_cblas.h>
#include <myarray.h>
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
bool diagonalize_deg(complex *S, double *e, int n, double degthr, double *eig, complex *v);
void Utrans(complex *u, complex *mat, int n);

void zhemm_interface(complex *C, bool left, complex *A, complex *B, int n, complex alpha = c1, complex beta = c0);

void zgemm_interface(complex *C, complex *A, complex *B, int n, complex alpha = c1, complex beta = c0, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans);
void zgemm_interface(complex *C, complex *A, complex *B, int m, int n, int k, complex alpha = c1, complex beta = c0, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans);

void aij_bji(double *C, complex *A, complex *B, int n, double alpha = 1, double beta = 0);

void mat_diag_mult(complex *C, complex *A, double *B, int n, complex alpha = c1, complex beta = c0);
void mat_diag_mult(complex *C, double *A, complex *B, int n, complex alpha = c1, complex beta = c0);
void vec3_dot_vec3array(complex *vm, vector3<double> v, complex **m, int n, complex b = c0, complex c = c0);
void vec3_dot_vec3array(complex *vm, vector3<complex> v, complex **m, int n, complex b = c0, complex c = c0); // m[3][]

// b[0:(i1-i0), 0:(j1-j0)] = a[i0:i1, j0:j1] of a[:, 0:n]
void trunc_copy_mat(complex *b, complex *a, int n, int i0, int i1, int j0, int j1);
complex* trunc_alloccopy_mat(complex* arr, int n, int i0, int i1, int j0, int j1);
// b[i0:i1, j0:j1] (of b[:, 0:n]) = a[0:(i1-i0), 0:(j1-j0)]
void set_mat(complex *b, complex *a, int n, int i0, int i1, int j0, int j1);
// notice that in the following two subroutines, complex arrays are actually array matrices
// take arr[:, bStart:bEnd, bStart:bEnd] from arr[:, 0:n2, 0:n2]
complex** trunc_alloccopy_arraymat(complex** arr, int n1, int n2, int bStart, int bEnd);
void trunc_copy_arraymat(complex** A, complex **B, int n1, int n2, int bStart, int bEnd);

void zeros_off_diag(complex* a, int n);
void zeros_off_diag(complex** arr, int n1, int n2);

void transpose(complex *m, complex *t, int n);
void transpose(complex *a, complex *t, int m, int n);
void hermite(complex *m, complex *h, int n);
void hermite(complex *a, complex *h, int m, int n);

void commutator_zhemm(complex *C, complex *A, complex *B, int n, complex alpha = c1);
void commutator_zgemm(complex *C, complex *A, complex *B, int n, complex alpha = c1); // seems not physical
void commutator_mat_diag(complex *C, complex *A, double *B, int n, complex alpha = c1);
void commutator_mat_diag(complex *C, double *A, complex *B, int n, complex alpha = c1);
double trace(complex *m, int n);
double trace_square(complex *m, int n);
double trace_square_hermite(complex *m, int n);
double trace(complex **m, int n1, int n);
double trace_square(complex **m, int n1, int n);
double trace_square_hermite(complex **m, int n1, int n);
double trace_AB(complex *A, complex *B, int n);
double trace_AB(complex *A, complex *B, int m, int n); // not implemented
double trace_AB(complex **A, complex **B, int n1, int n);
double trace_AB(complex **A, complex **B, int n1, int m, int n); // not implemented