#pragma once
#include <core/scalar.h>
#include <core/Units.h>
#include <core/matrix.h>
#include "FeynWann.h"

const complex c0(0, 0);
const complex c1(1, 0);
const complex cm1(-1, 0);
const complex ci(0, 1);
const complex cmi(0, -1);
const double bohr2cm = 5.291772109038e-9;
const double Tesla = eV*sec / (meter*meter);
const vector3<> K = vector3<>(1. / 3, 1. / 3, 0), Kp = vector3<>(-1. / 3, -1. / 3, 0);

template <typename T> int sgn(T val){
	return (T(0) < val) - (val < T(0));
}

vector3<> wrap_around_Gamma(const vector3<>& x){
	vector3<> result = x;
	for (int dir = 0; dir<3; dir++)
		result[dir] -= floor(0.5 + result[dir]);
	return result;
}
bool isKvalley(matrix3<> GGT, vector3<> k){
	return GGT.metric_length_squared(wrap_around_Gamma(K - k))
		< GGT.metric_length_squared(wrap_around_Gamma(Kp - k));
}
bool isInterValley(matrix3<> GGT, vector3<> k1, vector3<> k2){
	return isKvalley(GGT, k1) xor isKvalley(GGT, k2);
}

int dimension(FeynWann& fw);
double cell_size(FeynWann& fw);
void print_carrier_density(FeynWann& fw, double carrier_density);
double cminvdim2au(FeynWann& fw);
double find_mu(double ncarrier, double t, double mu0, std::vector<FeynWann::StateE>& e, int bStart, int bCBM, int bStop);
double compute_ncarrier(bool isHole, double t, double mu, std::vector<diagMatrix>& Ek, int nb, int nv);
double compute_ncarrier(bool isHole, double t, double mu, std::vector<FeynWann::StateE>& e, int bStart, int bCBM, int bStop);
std::vector<diagMatrix> computeF(double t, double mu, std::vector<FeynWann::StateE>& e, int bStart, int bStop);

vector3<> compute_bsq(std::vector<FeynWann::StateE>& e, int bStart, int bStop, double degthr, std::vector<diagMatrix> F);
matrix degProj(matrix& M, diagMatrix& E, double degthr);
void degProj(matrix& M, diagMatrix& E, double degthr, matrix& Mdeg);
double compute_sz(complex **dm, size_t nk, double nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e);
vector3<> compute_spin(std::vector<std::vector<matrix>> m, size_t nk, double nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e);
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
void axbyc(complex *y, complex *x, int n, complex a = c1, complex b = c0, complex c = c0); // y = ax + by + c, default = copy
void error_message(string s, string routine = "");
void printf_complex_mat(complex *m, int n, string s);
void fprintf_complex_mat(FILE *fp, complex *m, int n, string s);

double maxval(std::vector<FeynWann::StateE>& e, int bStart, int bStop){
	double r = DBL_MIN;
	for (size_t ik; ik < e.size(); ik++){
		diagMatrix Ek = e[ik].E(bStart, bStop);
		for (int b = 0; b < bStop - bStart; b++)
		if (Ek[b] > r) r = Ek[b];
	}
	return r;
}
double minval(std::vector<FeynWann::StateE>& e, int bStart, int bStop){
	double r = DBL_MAX;
	for (size_t ik; ik < e.size(); ik++){
		diagMatrix Ek = e[ik].E(bStart, bStop);
		for (int b = 0; b < bStop - bStart; b++)
		if (Ek[b] < r) r = Ek[b];
	}
	return r;
}

void check_file_size(FILE *fp, size_t expect_size, string message);
void fseek_bigfile(FILE *fp, size_t count, size_t size, int origin = SEEK_SET);

template <typename T>
void merge_files_mpi(string fname, T val, size_t n){
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