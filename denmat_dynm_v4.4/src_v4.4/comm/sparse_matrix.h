#pragma once
#include <scalar.h>
#include <constants.h>
#include <myarray.h>

struct sparse_mat{
	complex *s;
	int *i, *j;
	int ns;

	sparse_mat() : s(nullptr), i(nullptr), j(nullptr), ns(0) {}

	sparse_mat(int nsmax, bool alloc_only_s){
		s = new complex[nsmax]{c0};
		if (!alloc_only_s){
			i = new int[nsmax]();
			j = new int[nsmax]();
		}
		else{
			i = nullptr; j = nullptr;
		}
		ns = 0;
	}
};

void sparse_zgemm(complex *c, bool left, complex *s, int *indexi, int *indexj, int ns, complex *b, int m, int n, int k, complex alpha = c1, complex beta = c0);