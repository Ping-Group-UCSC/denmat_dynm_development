#pragma once
#include <core/scalar.h>
#include <core/Units.h>
#include <core/matrix.h>
#include <core/Random.h>
#include "FeynWann.h"

const complex c0(0, 0);
const complex c1(1, 0);
const complex cm1(-1, 0);
const complex ci(0, 1);
const complex cmi(0, -1);
const double bohr2cm = 5.291772109038e-9;
const double ps = 1e3*fs; //picosecond

vector3<> wrap_around_Gamma(const vector3<>& x);
vector3<> wrap(const vector3<>& x, vector3<> center = vector3<>(0,0,0));

template <typename T> int sgn(T val){
	return (T(0) < val) - (val < T(0));
}

matrix degProj(matrix& M, diagMatrix& E, double degthr);
void degProj(matrix& M, diagMatrix& E, double degthr, matrix& Mdeg);
void init_dm(complex **dm, size_t nk, int nb, std::vector<diagMatrix>& F);
void set_dm1(complex **dm, size_t nk, int nb, complex **dm1);

double** alloc_real_array(int n1, int n2, double val = 0.);
complex* alloc_array(int n1, complex val = c0);
complex** alloc_array(int n1, int n2, complex val = c0);
complex*** alloc_array(int n1, int n2, int n3, complex val = c0);

void zeros(double* arr, int n1);
void zeros(double** arr, int n1, int n2);
void zeros(complex* arr, int n1);
void zeros(complex** arr, int n1, int n2);
void zeros(complex*** arr, int n1, int n2, int n3);
void zeros(matrix& m);
void zeros(std::vector<matrix>& v);

void axbyc(double *y, double *x, size_t n, double a = 1, double b = 0, double c = 0); // y = ax + by + c, default = copy
void axbyc(double *y, double *x, int n, double a = 1, double b = 0, double c = 0); // y = ax + by + c, default = copy
void axbyc(double **y, double **x, int n1, int n2, double a = 1, double b = 0, double c = 0); // y = ax + by + c, default = copy
void axbyc(complex *y, complex *x, int n, complex a = c1, complex b = c0, complex c = c0); // y = ax + by + c, default = copy
void axbyc(complex **y, complex **x, int n1, int n2, complex a = c1, complex b = c0, complex c = c0); // y = ax + by + c, default = copy

double mean_of_array(double *a, int n, double *w = nullptr);
double sigma_of_array(double *a, int n, bool compute_mean = true, double mean_fixed = 0, double *w = nullptr);
void random_normal_array(double *a, int n, double mean = 0.0, double sigma = 1.0, double cap = 0.0, double *w = nullptr);
vector3<> mean_of_(std::vector<vector3<>> m, double *w = nullptr);
vector3<> sigma_of_(std::vector<vector3<>> m, bool compute_mean = true, vector3<> mean_fixed = vector3<>(), double *w = nullptr);
void random_normal_(std::vector<vector3<>>& m, vector3<> mean = vector3<>(), vector3<> sigma = vector3<>(1, 1, 1), vector3<> cap = vector3<>(), double *w = nullptr);
matrix3<> mean_of_(std::vector<matrix3<>> m, double *w = nullptr);
matrix3<> sigma_of_(std::vector<matrix3<>> m, bool compute_mean = true, matrix3<> mean_fixed = matrix3<>(), double *w = nullptr);
void random_normal_(std::vector<matrix3<>>& m, matrix3<> mean = matrix3<>(), matrix3<> sigma = matrix3<>(1, 1, 1), matrix3<> cap = matrix3<>(), double *w = nullptr);

void error_message(string s, string routine = "");
void printf_complex_mat(complex *m, int n, string s);
void fprintf_complex_mat(FILE *fp, complex *m, int n, string s);

void fprintf_real_array(FILE *fp, double *a, int n, string s, bool col_mode = false);
void fprintf_real_array(string fname, double *a, int n, string s, bool col_mode = false);
void fprintf_real_array(FILE *fp, double *a, int n, string s, string fmt_val, bool col_mode = false);
void fprintf_real_array(string fname, double *a, int n, string s, string fmt_val, bool col_mode = false);
void fprintf(FILE *fp, std::vector<vector3<>> m, string s, string fmt_val = " %lg");
void fprintf(string fname, std::vector<vector3<>> m, string s, string fmt_val = " %lg");
void fprintf(FILE *fp, std::vector<matrix3<>> m, string s, string fmt_val = " %lg");
void fprintf(string fname, std::vector<matrix3<>> m, string s, string fmt_val = " %lg");

bool exists(string name);
string int2str(int i);
size_t file_size(FILE *fp);
void check_file_size(FILE *fp, size_t expect_size, string message);
bool check_file_size(FILE *fp, size_t expect_size);
void fseek_bigfile(FILE *fp, size_t count, size_t size, int origin = SEEK_SET);

template <typename T>
void merge_files_mpi(string fname, T val, size_t n){
	if (mpiWorld->isHead()) system(("rm -rf " + fname).c_str()); MPI_Barrier(MPI_COMM_WORLD);

	string s = "Merge " + fname + "\n";
	logPrintf(s.c_str());

	std::vector<size_t> fcount(mpiWorld->nProcesses());
	if (mpiWorld->isHead()){
		for (int i = 0; i < mpiWorld->nProcesses(); i++){
			ostringstream convert; convert << i; convert.flush();
			string fnamei = fname + "." + convert.str();
			FILE *fpi = fopen(fnamei.c_str(), "rb");
			fseek(fpi, 0L, SEEK_END);
			fcount[i] = ftell(fpi) / sizeof(T);
			fclose(fpi);
		}
	}
	mpiWorld->bcastData(fcount);

	FILE *fpout = fopen(fname.c_str(), "wb");
	for (int i = 0; i < mpiWorld->iProcess(); i++)
		fseek_bigfile(fpout, fcount[i], sizeof(T), i == 0 ? SEEK_SET : SEEK_CUR);

	ostringstream convert; convert << mpiWorld->iProcess();
	convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise files are not created for non-root processes
	string fnamein = fname + "." + convert.str();
	FILE *fpin = fopen(fnamein.c_str(), "rb");

	std::vector<T> work(n);
	while (fread(work.data(), sizeof(T), n, fpin) == n)
		fwrite(work.data(), sizeof(T), n, fpout);

	fclose(fpin); fclose(fpout);
	remove(fnamein.c_str());
}
template <typename T>
void merge_files(string fname, T val, size_t n){
	MPI_Barrier(MPI_COMM_WORLD);
	if (mpiWorld->isHead()){
		if (mpiWorld->isHead()) system(("rm -rf " + fname).c_str());
		string s = "Merge " + fname + ":\nprocessing file ";
		logPrintf(s.c_str());
		std::vector<T> work(n);
		for (int i = 0; i < mpiWorld->nProcesses(); i++){
			ostringstream convert; convert << i;
			read_append_file(fname + "." + convert.str(), fname, work, n);
			if (i % 10 == 0) printf("%d ", i);
		}
		logPrintf("done\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
template <typename T>
size_t read_append_file(string fnamein, string fnameout, std::vector<T>& v, size_t n){
	FILE *fpin = fopen(fnamein.c_str(), "rb");
	FILE *fpout = fopen(fnameout.c_str(), "ab");
	size_t nline = 0;
	while (fread(v.data(), sizeof(T), n, fpin) == n){
		nline++;
		fwrite(v.data(), sizeof(T), n, fpout);
	}
	fclose(fpin); fclose(fpout);
	remove(fnamein.c_str());
	return nline;
}