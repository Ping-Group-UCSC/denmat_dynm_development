#include <mymatrix.h>
#include <stdio.h>
#include <stdexcept>

void diagonalize(complex *H, int n, double *eig, complex *v){
	char jobz = 'V'; //compute eigenvectors and eigenvalues
	char range = 'A'; //compute all eigenvalues
	char uplo = 'U'; //use upper-triangular part
	double eigMin = 0., eigMax = 0.; //eigenvalue range (not used for range-type 'A')
	int indexMin = 0, indexMax = 0; //eignevalue index range (not used for range-type 'A')
	double absTol = 0.;
	int nEigsFound;
	int iSuppz[2 * n];
	int lwork = (64 + 1)*n; complex work[lwork]; //Magic number 64 obtained by running ILAENV as suggested in doc of zheevr (and taking the max over all N)
	int lrwork = 24 * n; double rwork[lrwork]; //from doc of zheevr
	int liwork = 10 * n; int iwork[liwork]; //from doc of zheevr
	int info = 0;

	complex a[n*n];
	// a will be destroyed
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
		a[i*n + j] = H[j*n + i]; // From RowMajor to ColMajor
	complex vCol[n*n];

	zheevr_(&jobz, &range, &uplo, &n, a, &n, &eigMin, &eigMax, &indexMin, &indexMax, &absTol, &nEigsFound, eig, vCol, &n,
		iSuppz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
	// notice that the output eigenvectors are Column Major
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
		v[i*n + j] = vCol[j*n + i];
	if (info<0)
		printf("Argument# %d to LAPACK eigenvalue routine ZHEEVR is invalid.\n", -info);
	if (info>0)
		printf("Error code %d in LAPACK eigenvalue routine ZHEEVR.\n", info);
}

void zhemm_interface(complex *C, bool left, complex *A, complex *B, int n, complex alpha, complex beta){
	CBLAS_SIDE side = left ? CblasLeft : CblasRight;
	cblas_zhemm(CblasRowMajor, side, CblasUpper, n, n, &alpha, A, n, B, n, &beta, C, n);
}

void zgemm_interface(complex *C, complex *A, complex *B, int n, complex alpha, complex beta, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB){
	cblas_zgemm(CblasRowMajor, transA, transB, n, n, n, &alpha, A, n, B, n, &beta, C, n);
}

void zgemm_interface(complex *C, complex *A, complex *B, int m, int n, int k, complex alpha, complex beta, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB){
	int lda = (transA == CblasNoTrans) ? k : m;
	int ldb = (transB == CblasNoTrans) ? n : k;
	cblas_zgemm(CblasRowMajor, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, n);
}

void aij_bji(double *C, complex *A, complex *B, int n, double alpha, double beta){
	for (int i = 0; i < n; i++)
		C[i] *= beta;
	if (alpha != 0){
		for (int i = 0; i < n; i++){
			double tmp = 0;
			for (int j = 0; j < n; j++)
				tmp += real(A[i*n + j] * B[j*n + i]);
			C[i] += alpha * tmp;
		}
	}
}

void trunc_copy_mat(complex *b, complex *a, int n, int i0, int i1, int j0, int j1){
// b[0:(i1-i0), 0:(j1-j0)] = a[i0:i1, j0:j1] of a[:, 0:n]
	for (int i = 0; i < i1 - i0; i++)
	for (int j = 0; j < j1 - j0; j++)
		b[i*(j1 - j0) + j] = a[(i + i0)*n + j + j0];
}
void set_mat(complex *b, complex *a, int n, int i0, int i1, int j0, int j1){
// b[i0:i1, j0:j1] (of b[:, 0:n]) = a[0:(i1-i0), 0:(j1-j0)]
	for (int i = i0; i < i1; i++)
	for (int j = i0; j < j1; j++)
		b[i*n + j] = a[(i - i0)*(j1-j0) + j - j0];
}
void hermite(complex *m, complex *h, int n){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
		h[i*n + j] = conj(m[j*n + i]);
}
void hermite(complex *a, complex *h, int m, int n){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < m; j++)
		h[i*m + j] = conj(a[j*n + i]);
}

void mat_diag_mult(complex *C, complex *A, double *B, int n){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
		C[i*n + j] = A[i*n + j] * B[j];
}
void mat_diag_mult(complex *C, double *A, complex *B, int n){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
		C[i*n + j] = A[i] * B[i*n + j];
}

void vec3_dot_vec3array(complex *vm, vector3<double> v, complex **m, int n){
	for (int i = 0; i < n; i++)
		vm[i] = v[0] * m[0][i] + v[1] * m[1][i] + v[2] * m[2][i];
}
void vec3_dot_vec3array(complex *vm, vector3<complex> v, complex **m, int n){
	for (int i = 0; i < n; i++)
		vm[i] = v[0] * m[0][i] + v[1] * m[1][i] + v[2] * m[2][i];
}