#pragma once
#include <scalar.h>
#include <vector3.h>
#include <matrix3.h>
#include <constants.h>
#include <Random.h>
#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

using namespace std;

template <typename T>
vector<int> sort_indexes(const vector<T> &v) {

	// initialize original index locations
	vector<int> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	stable_sort(idx.begin(), idx.end(),
		[&v](int i1, int i2) {return v[i1] < v[i2]; });

	return idx;
}

double** alloc_real_array(int n1, int n2, double val = 0.);
double*** alloc_real_array(int n1, int n2, int n3, double val = 0.);
complex** alloc_array(int n1, int n2, complex val = c0);
complex*** alloc_array(int n1, int n2, int n3, complex val = c0);
complex**** alloc_array(int n1, int n2, int n3, int n4, complex val = c0);
void dealloc_real_array(double**& arr);
void dealloc_real_array(double***& arr);
void dealloc_array(complex**& arr);
void dealloc_array(complex***& arr);

double** trunc_alloccopy_array(double** arr, int n1, int n2_start, int n2_end);
void trunc_copy_array(double** A, double **B, int n1, int n2_start, int n2_end);

void zeros(double* arr, int n1);
void zeros(double** arr, int n1, int n2);
void zeros(double*** arr, int n1, int n2, int n3);
void zeros(complex* arr, int n1);
void zeros(complex** arr, int n1, int n2);
void zeros(complex*** arr, int n1, int n2, int n3);
void zeros(complex**** arr, int n1, int n2, int n3, int n4);

void random_array(complex* a, int n);
double mean_of_array(double *a, int n, double *w = nullptr);
double sigma_of_array(double *a, int n, bool compute_mean = true, double mean_fixed = 0, double *w = nullptr);
void random_normal_array(double *a, int n, double mean = 0.0, double sigma = 1.0, double cap = 0.0, double *w = nullptr);
vector3<> mean_of_(std::vector<vector3<>> m, double *w = nullptr);
vector3<> sigma_of_(std::vector<vector3<>> m, bool compute_mean = true, vector3<> mean_fixed = vector3<>(), double *w = nullptr);
void random_normal_(std::vector<vector3<>>& m, vector3<> mean = vector3<>(), vector3<> sigma = vector3<>(1, 1, 1), vector3<> cap = vector3<>(), double *w = nullptr);
matrix3<> mean_of_(std::vector<matrix3<>> m, double *w = nullptr);
matrix3<> sigma_of_(std::vector<matrix3<>> m, bool compute_mean = true, matrix3<> mean_fixed = matrix3<>(), double *w = nullptr);
void random_normal_(std::vector<matrix3<>>& m, matrix3<> mean = matrix3<>(), matrix3<> sigma = matrix3<>(1, 1, 1), matrix3<> cap = matrix3<>(), double *w = nullptr);

void conj(complex* a, complex* c, int n1);

double maxval(double *arr, int n1);
double maxval(double **arr, int n1, int bStart, int bEnd);
double minval(double *arr, int n1);
double minval(double **arr, int n1, int bStart, int bEnd);

double dot(double *v1, double *v2, int n);
void axbyc(double *y, double *x, size_t n, double a = 1, double b = 0, double c = 0); // y = ax + by + c, default = copy
void axbyc(double *y, double *x, int n, double a = 1, double b = 0, double c = 0); // y = ax + by + c, default = copy
void axbyc(double **y, double **x, int n1, int n2, double a = 1, double b = 0, double c = 0); // y = ax + by + c, default = copy
void axbyc(double ***y, double ***x, int n1, int n2, int n3, double a = 1, double b = 0, double c = 0); // y = ax + by + c, default = copy
void axbyc(complex *y, complex *x, int n, complex a = c1, complex b = c0, complex c = c0); // y = ax + by + c, default = copy
void axbyc(complex **y, complex **x, int n1, int n2, complex a = c1, complex b = c0, complex c = c0); // y = ax + by + c, default = copy