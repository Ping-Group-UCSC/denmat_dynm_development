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
bool diagonalize_deg(complex *S, double *e, int n, double degthr, double *eig, complex *v){
	bool isI = true;
	zeros(v, n*n);
	for (int i = 0; i < n; i++){ v[i*n + i] = c1; eig[i] = real(S[i*n + i]); }
	int i = 0, j;
	while (i < n){
		j = i + 1;
		for (; j < n; j++){ if (fabs(e[j] - e[i]) >= degthr) break; }
		int n_block = j - i;
		if (n_block > 1){
			isI = false;

			complex *S_block = trunc_alloccopy_mat(S, n, i, j, i, j);
			double *eig_block = new double[n_block]{0};
			complex *v_block = new complex[n_block*n_block]{c0};

			diagonalize(S_block, n_block, eig_block, v_block);

			for (int b = i; b < j; b++){ eig[b] = eig_block[b - i]; }
			set_mat(v, v_block, n, i, j, i, j); // b[i0:i1, j0:j1] (of b[:, 0:n]) = a[0:(i1-i0), 0:(j1-j0)]

			delete[] S_block; delete[] eig_block; delete[] v_block;
		}
		i = j;
	}
	return isI;
}
void Utrans(complex *u, complex *mat, int n){
	// u^dagger * mat * u
	complex *uh = new complex[n*n]{c0};
	hermite(u, uh, n);
	complex *mtmp = new complex[n*n]{c0};
	zhemm_interface(mtmp, true, mat, u, n);
	zgemm_interface(mat, uh, mtmp, n);
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
complex* trunc_alloccopy_mat(complex* arr, int n, int i0, int i1, int j0, int j1){
	complex* r = new complex[(i1 - i0)*(j1 - j0)]{c0};
	for (int i = 0; i < i1 - i0; i++)
	for (int j = 0; j < j1 - j0; j++)
		r[i*(j1 - j0) + j] = arr[(i + i0)*n + j + j0];
	return r;
}
void set_mat(complex *b, complex *a, int n, int i0, int i1, int j0, int j1){
// b[i0:i1, j0:j1] (of b[:, 0:n]) = a[0:(i1-i0), 0:(j1-j0)]
	for (int i = i0; i < i1; i++)
	for (int j = i0; j < j1; j++)
		b[i*n + j] = a[(i - i0)*(j1-j0) + j - j0];
}
// notice that in the following two subroutines, complex arrays are actually array matrices
complex** trunc_alloccopy_arraymat(complex** arr, int n1, int n2, int bStart, int bEnd){
	// take arr[:, bStart:bEnd, bStart:bEnd] from arr[:, 0:n2, 0:n2]
	int nb = bEnd - bStart;
	complex** r = alloc_array(n1, nb*nb);
	for (int i1 = 0; i1 < n1; i1++)
	for (int b1 = 0; b1 < nb; b1++)
	for (int b2 = 0; b2 < nb; b2++){
		int b1_full = b1 + bStart;
		int b2_full = b2 + bStart;
		r[i1][b1*nb + b2] = arr[i1][b1_full*n2 + b2_full];
	}
	return r;
}
void trunc_copy_arraymat(complex** A, complex **B, int n1, int n2, int bStart, int bEnd){
	// take A[0:n1, bStart:bEnd, bStart:bEnd] from B[0:n1, 0:n2, 0:n2]
	int nb = bEnd - bStart;
	for (int i1 = 0; i1 < n1; i1++)
	for (int b1 = 0; b1 < nb; b1++)
	for (int b2 = 0; b2 < nb; b2++){
		int b1_full = b1 + bStart;
		int b2_full = b2 + bStart;
		A[i1][b1*nb + b2] = B[i1][b1_full*n2 + b2_full];
	}
}

void zeros_off_diag(complex* a, int n){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
		if (i != j) a[i*n + j] = c0;
}
void zeros_off_diag(complex** arr, int n1, int n2){
	for (int i1 = 0; i1 < n1; i1++)
		zeros_off_diag(arr[i1], n2);
}

void transpose(complex *m, complex *t, int n){
	transpose(m, t, n, n);
}
void transpose(complex *a, complex *t, int m, int n){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < m; j++)
		t[i*m + j] = a[j*n + i];
}
void hermite(complex *m, complex *h, int n){
	hermite(m, h, n, n);
}
void hermite(complex *a, complex *h, int m, int n){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < m; j++)
		h[i*m + j] = conj(a[j*n + i]);
}

void mat_diag_mult(complex *C, complex *A, double *B, int n, complex alpha, complex beta){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			if (beta.real() == 0 && beta.imag() == 0)
				C[i*n + j] = c0;
			else
				C[i*n + j] *= beta;
			if (alpha.real() == 1 && alpha.imag() == 0)
				C[i*n + j] += A[i*n + j] * B[j];
			else
				C[i*n + j] += alpha * A[i*n + j] * B[j]; // not very optimized, we can first multiply alpha with diagonal matrix B
		}
	}
}
void mat_diag_mult(complex *C, double *A, complex *B, int n, complex alpha, complex beta){
	for (int i = 0; i < n; i++){
		complex ctmp = A[i];
		if (alpha.real() != 1 || alpha.imag() != 0)
			ctmp *= alpha;
		for (int j = 0; j < n; j++){
			if (beta.real() == 0 && beta.imag() == 0)
				C[i*n + j] = c0;
			else
				C[i*n + j] *= beta;
			C[i*n + j] += ctmp * B[i*n + j];
		}
	}
}

void vec3_dot_vec3array(complex *vm, vector3<double> v, complex **m, int n, complex b, complex c){
	axbyc(vm, m[0], n, complex(v[0],0), b, c);
	axbyc(vm, m[1], n, complex(v[1],0), c1);
	axbyc(vm, m[2], n, complex(v[2],0), c1);
	//for (int i = 0; i < n; i++)
	//	vm[i] = v[0] * m[0][i] + v[1] * m[1][i] + v[2] * m[2][i];
}
void vec3_dot_vec3array(complex *vm, vector3<complex> v, complex **m, int n, complex b, complex c){
	axbyc(vm, m[0], n, v[0], b, c);
	axbyc(vm, m[1], n, v[1], c1);
	axbyc(vm, m[2], n, v[2], c1);
	//for (int i = 0; i < n; i++)
	//	vm[i] = v[0] * m[0][i] + v[1] * m[1][i] + v[2] * m[2][i];
}

void commutator_zhemm(complex *C, complex *A, complex *B, int n, complex alpha){
	// A * B - B * A
	zhemm_interface(C, false, A, B, n);
	zhemm_interface(C, true, A, B, n, c1, cm1);
	for (int i = 0; i < n*n; i++)
		C[i] *= alpha;
}
void commutator_mat_diag(complex *C, complex *A, double *B, int n, complex alpha){
	mat_diag_mult(C, B, A, n);
	mat_diag_mult(C, A, B, n, c1, cm1);
	for (int i = 0; i < n*n; i++)
		C[i] *= alpha;
}
void commutator_mat_diag(complex *C, double *A, complex *B, int n, complex alpha){
	mat_diag_mult(C, B, A, n);
	mat_diag_mult(C, A, B, n, c1, cm1);
	for (int i = 0; i < n*n; i++)
		C[i] *= alpha;
}
double trace(complex *m, int n){
	double r = 0;
	for (int i = 0; i < n; i++)
		r += real(m[i*n + i]);
	return r;
}
double trace_square(complex *m, int n){
	double r = 0;
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
		r += real(m[i*n + j] * m[j*n + i]);
	return r;
}
double trace_square_hermite(complex *m, int n){
	double r = 0;
	for (int i = 0; i < n; i++){
		r += m[i*n + i].norm();
		for (int j = i + 1; j < n; j++)
			r += 2 * m[i*n + j].norm();
	}
	return r;
}
double trace(complex **m, int n1, int n){
	double r = 0;
	for (int i = 0; i < n1; i++)
		r += trace(m[i], n);
	return r;
}
double trace_square(complex **m, int n1, int n){
	double r = 0;
	for (int i = 0; i < n1; i++)
		r += trace_square(m[i], n);
	return r;
}
double trace_square_hermite(complex **m, int n1, int n){
	double r = 0;
	for (int i = 0; i < n1; i++)
		r += trace_square_hermite(m[i], n);
	return r;
}
double trace_AB(complex *A, complex *B, int n){
	double r = 0;
	double *C = new double[n]{0};
	aij_bji(C, A, B, n, 1, 0);
	for (int i = 0; i < n; i++)
		r += C[i];
	return r;
}
double trace_AB(complex **A, complex **B, int n1, int n){
	double r = 0;
	for (int i = 0; i < n1; i++)
		r += trace_AB(A[i], B[i], n);
	return r;
}