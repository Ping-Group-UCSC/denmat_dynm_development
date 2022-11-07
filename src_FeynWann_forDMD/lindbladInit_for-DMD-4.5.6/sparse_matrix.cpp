#include "sparse_matrix.h"

void sparse_plus_dense(sparse_mat *smat, double thrsparse, complex *m, int ni, int nj, complex a, complex b, complex c){ // s = am + bs + c, default = copy
	complex *dense = smat->todense(ni, nj);
	axbyc(dense, m, ni*nj, a, b, c);
	smat->sparse(dense, ni, nj, thrsparse);
	delete[] dense;
}