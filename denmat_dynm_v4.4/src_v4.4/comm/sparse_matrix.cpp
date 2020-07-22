#include <sparse_matrix.h>

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