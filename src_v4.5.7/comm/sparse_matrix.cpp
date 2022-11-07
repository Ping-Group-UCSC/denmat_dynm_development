#include <sparse_matrix.h>

void sparse_zgemm(complex *c, bool left, sparse_mat *smat, complex *b, int m, int n, int k, complex alpha, complex beta){
	sparse_zgemm(c, left, smat->s, smat->i, smat->j, smat->ns, b, m, n, k, alpha, beta);
}
void sparse_zgemm(complex *c, bool left, complex *s, int *indexi, int *indexj, int ns, complex *b, int m, int n, int k, complex alpha, complex beta){
	if (beta.real() == 0 && beta.imag() == 0)
		zeros(c, m*n);
	else{
		for (int i = 0; i < m*n; i++)
			c[i] *= beta;
	}

	if (alpha.real() != 0 || alpha.imag() != 0){
		complex as[ns];
		if (alpha.real() == 1 && alpha.imag() == 0)
			for (int is = 0; is < ns; is++)
				as[is] = s[is];
		else
			for (int is = 0; is < ns; is++)
				as[is] = alpha * s[is];

		if (left){
			for (int is = 0; is < ns; is++){
				int i = indexi[is];
				int i2 = indexj[is];
				for (int j = 0; j < n; j++)
					c[i*n + j] += as[is] * b[i2*n + j];
			}
		}
		else{
			for (int is = 0; is < ns; is++){
				int i2 = indexi[is];
				int j = indexj[is];
				for (int i = 0; i < m; i++)
					c[i*n + j] += b[i*k + i2] * as[is];
			}
		}
	}
}

void sparse_plus_dense(sparse_mat *smat, double thrsparse, complex *m, int ni, int nj, complex a, complex b, complex c){ // s = am + bs + c, default = copy
	complex *dense = smat->todense(ni, nj);
	axbyc(dense, m, ni*nj, a, b, c);
	smat->sparse(dense, ni, nj, thrsparse);
	delete[] dense;
}

void sparse_plus_dense(sparse_mat **smat, double thrsparse, complex **m, int nk, int ni, int nj, complex a, complex b, complex c){ // s = am + bs + c, default = copy
	for (int ik = 0; ik < nk; ik++)
		sparse_plus_dense(smat[ik], thrsparse, m == nullptr ? nullptr : m[ik], ni, nj, a, b, c);
}