#include <myarray.h>
#include <stdio.h>
#include <float.h>

void axbyc(double *y, double *x, size_t n, double a, double b, double c){
	if (b == 0) zeros(y, n);
	else if (b != 1) for (size_t i = 0; i < n; i++) { y[i] *= b; }
	if (x == nullptr || a == 0) for (size_t i = 0; i < n; i++){ y[i] += c; }
	else if (a == 1) for (size_t i = 0; i < n; i++) { y[i] += x[i] + c; }
	else for (size_t i = 0; i < n; i++) { y[i] += a * x[i] + c; }
}
void axbyc(double *y, double *x, int n, double a, double b, double c){
	axbyc(y, x, (size_t)n, a, b, c);
}
void axbyc(double **y, double **x, int n1, int n2, double a, double b, double c){
	if (b == 0)
		zeros(y, n1, n2);
	else if (b != 1){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] *= b;
	}
	if (x == nullptr || a == 0){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += c;
	}
	else if (a == 1){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += x[i1][i2] + c;
	}
	else{
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += a * x[i1][i2] + c;
	}
}
void axbyc(complex **y, complex **x, int n1, int n2, complex a, complex b, complex c){
	if (b.real() == 0 && b.imag() == 0)
		zeros(y, n1, n2);
	else if (!(b.real() == 1 && b.imag() == 0)){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] *= b;
	}
	if (x == nullptr || (a.real() == 0 && a.imag() == 0)){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += c;
	}
	else if (a.real() == 1 && a.imag() == 0){
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += x[i1][i2] + c;
	}
	else{
		for (int i1 = 0; i1 < n1; i1++)
		for (int i2 = 0; i2 < n2; i2++)
			y[i1][i2] += a * x[i1][i2] + c;
	}
}
void axbyc(complex *y, complex *x, int n, complex a, complex b, complex c){
	if (b.real() == 0 && b.imag() == 0) zeros(y, n);
	else if (!(b.real() == 1 && b.imag() == 0))
		for (int i = 0; i < n; i++){ y[i] *= b; }
	if (x == nullptr || (a.real() == 0 && a.imag() == 0))
		for (int i = 0; i < n; i++){ y[i] += c; }
	else if (a.real() == 1 && a.imag() == 0)
		for (int i = 0; i < n; i++){ y[i] += x[i] + c; }
	else
		for (int i = 0; i < n; i++){ y[i] += a * x[i] + c; }
}

double dot(double *v1, double *v2, int n){
	double result = 0;
	for (int i = 0; i < n; i++)
		result += v1[i] * v2[i];
	return result;
}

double** trunc_alloccopy_array(double** arr, int n1, int n2_start, int n2_end){
	double** r = alloc_real_array(n1, n2_end - n2_start);
	for (int i1 = 0; i1 < n1; i1++)
	for (int i2 = 0; i2 < n2_end - n2_start; i2++)
		r[i1][i2] = arr[i1][i2 + n2_start];
	return r;
}
void trunc_copy_array(double** A, double **B, int n1, int n2_start, int n2_end){
	for (int i1 = 0; i1 < n1; i1++)
	for (int i2 = 0; i2 < n2_end - n2_start; i2++)
		A[i1][i2] = B[i1][i2 + n2_start];
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
complex**** alloc_array(int n1, int n2, int n3, int n4, complex val){
	complex**** arr;
	if (n1 == 0) return arr;
	try{ arr = new complex***[n1]; }
	catch (std::bad_alloc& ba){ printf("bad_alloc of arr caught: %s in alloc_array(n1,n2,n3,n4)\n", ba.what()); }
	for (int i1 = 0; i1 < n1; i1++)
		arr[i1] = alloc_array(n2, n3, n4, val);
	return arr;
}
void dealloc_real_array(double**& arr){
	delete[] arr[0];  // remove the pool
	delete[] arr;     // remove the pointers
	arr = nullptr;
}
void dealloc_array(complex**& arr){
	delete[] arr[0];  // remove the pool
	delete[] arr;     // remove the pointers
	arr = nullptr;
}

double maxval(double *arr, int bStart, int bEnd){
	double r = DBL_MIN;
	for (int i1 = bStart; i1 < bEnd; i1++)
		if (arr[i1] > r)
			r = arr[i1];
	return r;
}
double maxval(double **arr, int n1, int bStart, int bEnd){
	double r = DBL_MIN;
	for (int i1 = 0; i1 < n1; i1++)
	for (int i2 = bStart; i2 < bEnd; i2++)
		if (arr[i1][i2] > r)
			r = arr[i1][i2];
	return r;
}
double minval(double *arr, int bStart, int bEnd){
	double r = DBL_MAX;
	for (int i1 = bStart; i1 < bEnd; i1++)
	if (arr[i1] < r)
		r = arr[i1];
	return r;
}
double minval(double **arr, int n1, int bStart, int bEnd){
	double r = DBL_MAX;
	for (int i1 = 0; i1 < n1; i1++)
	for (int i2 = bStart; i2 < bEnd; i2++)
		if (arr[i1][i2] < r)
			r = arr[i1][i2];
	return r;
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
void zeros(complex**** arr, int n1, int n2, int n3, int n4){
	for (int i1 = 0; i1 < n1; i1++)
		zeros(arr[i1], n2, n3, n4);
}
void random_array(complex* a, int n1){
	for (int i1 = 0; i1 < n1; i1++)
		a[i1] = complex(Random::uniform(), Random::uniform());
}
void conj(complex* a, complex* c, int n1){
	for (int i1 = 0; i1 < n1; i1++)
		c[i1] = conj(a[i1]);
}