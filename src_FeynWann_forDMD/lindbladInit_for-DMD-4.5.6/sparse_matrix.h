#pragma once
#include <core/scalar.h>
#include "help.h"
#include <stdio.h>

struct sparse_mat{
	complex *s;
	int *i, *j;
	int ns;

	sparse_mat() : s(nullptr), i(nullptr), j(nullptr), ns(0) {}

	sparse_mat(int nsmax, bool alloc_only_s) : sparse_mat() {
		alloc(nsmax, alloc_only_s);
	}
	void alloc(int nsmax, bool alloc_only_s){
		del();
		if (nsmax == 0) return;
		try { s = new complex[nsmax]{c0}; }
		catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in allocate s\n", ba.what()); }
		if (!alloc_only_s){
			try{ i = new int[nsmax]{-1}; }
			catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in allocate i\n", ba.what()); }
			try{ j = new int[nsmax]{-1}; }
			catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in allocate j\n", ba.what()); }
		}
	}

	sparse_mat(FILE *fpns, FILE *fps, FILE *fpi, FILE *fpj) : sparse_mat() {
		fread(&ns, sizeof(int), 1, fpns);
		alloc(ns, false);
		fread(s, 2*sizeof(double), ns, fps);
		fread(i, sizeof(int), ns, fpi);
		fread(j, sizeof(int), ns, fpj);
	}
	void write_to_files(FILE *fpns, FILE *fps, FILE *fpi, FILE *fpj){
		fwrite(&ns, sizeof(int), 1, fpns);
		fwrite(s, 2 * sizeof(double), ns, fps);
		fwrite(i, sizeof(int), ns, fpi);
		fwrite(j, sizeof(int), ns, fpj);
	}

	void del(){
		delete[] s; delete[] i; delete[] j;
		s = nullptr; i = nullptr; j = nullptr;
 	}
	~sparse_mat(){ del(); }

	sparse_mat(complex* A, int ni, double thr = 1e-40) : sparse_mat() {
		sparse(A, ni, thr);
	}
	sparse_mat(complex* A, int ni, int nj, double thr = 1e-40) : sparse_mat() {
		sparse(A, ni, nj, thr);
	}
	void sparse(complex* A, int ni, double thr=1e-40){ sparse(A, ni, ni, thr); }
	void sparse(complex* A, int ni, int nj, double thr=1e-40){
		get_ns(A, ni*nj, thr);
		alloc(this->ns, false);
		int is = 0;
		for (int i = 0; i < ni; i++)
		for (int j = 0; j < nj; j++)
		if (abs(A[i*nj + j]) > thr){
			this->s[is] = A[i*nj + j];
			this->i[is] = i;
			this->j[is] = j;
			is++;
		}
	}
	void get_ns(complex* A, int nij, double thr=1e-40){
		ns = 0;
		for (int ij = 0; ij < nij; ij++)
			if (abs(A[ij]) > thr) ns++;
	}

	void todense(complex *A, int ni, int nj){
		zeros(A, ni*nj);
		for (int is = 0; is < ns; is++)
			A[i[is]*nj + j[is]] = s[is];
	}
	complex *todense(int ni, int nj){
		complex *A = new complex[ni*nj]{c0};
		for (int is = 0; is < ns; is++)
			A[i[is] * nj + j[is]] = s[is];
		return A;
	}
};

void sparse_zgemm(complex *c, bool left, sparse_mat *smat, complex *b, int m, int n, int k, complex alpha = c1, complex beta = c0);
void sparse_zgemm(complex *c, bool left, complex *s, int *indexi, int *indexj, int ns, complex *b, int m, int n, int k, complex alpha = c1, complex beta = c0);
void sparse_plus_dense(sparse_mat *smat, double thrsparse, complex *m, int ni, int nj, complex a = c1, complex b = c0, complex c = c0); // s = am + bs + c, default = copy