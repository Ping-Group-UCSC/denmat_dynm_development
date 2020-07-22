#include "help_lindbladInit_for-DMD-4.4.h"

double compute_sz(complex **dm, size_t nk, size_t nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e){
	double result = 0.;
	for (size_t ik = 0; ik < nk; ik++){
		matrix s = e[ik].S[2](bStart, bStop, bStart, bStop);
		for (int b2 = 0; b2 < nb; b2++)
		for (int b1 = 0; b1 < nb; b1++)
			result += real(s(b1, b2) * dm[ik][b2*nb + b1]);
	}
	return result / nkTot;
}
vector3<> compute_spin(std::vector<std::vector<matrix>> m, size_t nk, size_t nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e){
	vector3<> result(0., 0., 0.);
	for (size_t ik = 0; ik < nk; ik++)
	for (int id = 0; id < 3; id++){
		matrix s = e[ik].S[id](bStart, bStop, bStart, bStop);
		for (int b2 = 0; b2 < nb; b2++)
		for (int b1 = 0; b1 < nb; b1++)
			result[id] += real(s(b1, b2) * m[ik][id](b2, b1));
	}
	return result / nkTot;
}
void init_dm(complex **dm, size_t nk, int nb, std::vector<diagMatrix>& F){
	for (size_t ik = 0; ik < nk; ik++)
	for (int b1 = 0; b1 < nb; b1++)
	for (int b2 = 0; b2 < nb; b2++)
	if (b1 == b2)
		dm[ik][b1*nb + b2] = F[ik][b1];
	else
		dm[ik][b1*nb + b2] = c0;
}
void set_dm1(complex **dm, size_t nk, int nb, complex **dm1){
	for (size_t ik = 0; ik < nk; ik++)
	for (int b1 = 0; b1 < nb; b1++)
	for (int b2 = 0; b2 < nb; b2++)
	if (b1 == b2)
		dm1[ik][b1*nb + b2] = c1 - dm[ik][b1*nb + b2];
	else
		dm1[ik][b1*nb + b2] = -dm[ik][b1*nb + b2];
}

double** alloc_real_array(int n1, int n2, double val){
	double** ptr = nullptr;
	double* pool = nullptr;
	if (n1 * n2 == 0) return ptr;
	try{
		ptr = new double*[n1];  // allocate pointers (can throw here)
		pool = new double[n1*n2]{val};  // allocate pool (can throw here)
		for (int i = 0; i < n1; i++, pool += n2)
			ptr[i] = pool; // now point the row pointers to the appropriate positions in the memory pool
		return ptr;
	}
	catch (std::bad_alloc& ex){ delete[] ptr; throw ex; }
}
complex* alloc_array(int n1, complex val){
	complex* arr;
	if (n1 == 0) return arr;
	try{ arr = new complex[n1]{val}; }
	catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in alloc_array(n1)\n", ba.what()); }
	return arr;
}
complex** alloc_array(int n1, int n2, complex val){
	complex** ptr = nullptr;
	complex* pool = nullptr;
	if (n1 * n2 == 0) return ptr;
	try{
		ptr = new complex*[n1];  // allocate pointers (can throw here)
		pool = new complex[n1*n2]{val};  // allocate pool (can throw here)
		for (int i = 0; i < n1; i++, pool += n2)
			ptr[i] = pool; // now point the row pointers to the appropriate positions in the memory pool
		return ptr;
	}
	catch (std::bad_alloc& ex){ delete[] ptr; throw ex; }
}
complex*** alloc_array(int n1, int n2, int n3, complex val){
	complex*** arr;
	if (n1 == 0) return arr;
	try{ arr = new complex**[n1]; }
	catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in alloc_array(n1,n2,n3)\n", ba.what()); }
	for (int i1 = 0; i1 < n1; i1++)
		arr[i1] = alloc_array(n2, n3, val);
	return arr;
}

void zeros(double* arr, int n1){
	for (int i1 = 0; i1 < n1; i1++)
		arr[i1] = 0.;
}
void zeros(double** arr, int n1, int n2){
	for (int i1 = 0; i1 < n1; i1++)
		zeros(arr[i1], n2);
}
void zeros(complex* arr, int n1){
	for (int i1 = 0; i1 < n1; i1++)
		arr[i1] = c0;
}
void zeros(complex** arr, int n1, int n2){
	for (int i1 = 0; i1 < n1; i1++)
		zeros(arr[i1], n2);
}
void zeros(complex*** arr, int n1, int n2, int n3){
	for (int i1 = 0; i1 < n1; i1++)
		zeros(arr[i1], n2, n3);
}
void zeros(matrix& m, int n){
	complex *mData = m.data();
	for (int i = 0; i < n*n; i++)
		*(mData++) = complex(0, 0);
}
void zeros(std::vector<matrix>& v, int n){
	for (matrix& m : v)
		zeros(m, n);
}
void error_message(string s, string routine){
	printf((s + " in " + routine).c_str());
	exit(EXIT_FAILURE);
}