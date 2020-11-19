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
double find_mu(double ncarrier, double t, double mu0, std::vector<FeynWann::StateE>& e, int bStart, int bStop);
double compute_ncarrier(bool isHole, double t, double mu, std::vector<diagMatrix>& Ek, int nb);
double compute_ncarrier(bool isHole, double t, double mu, std::vector<FeynWann::StateE>& e, int bStart, int bStop);
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
void error_message(string s, string routine);
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