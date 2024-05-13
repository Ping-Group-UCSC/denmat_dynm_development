#pragma once
#include "common_headers.h"

double compute_sz(complex **dm, size_t nk, double nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e);
vector3<> compute_spin(std::vector<std::vector<matrix>> m, size_t nk, double nkTot, int nb, int bStart, int bStop, std::vector<FeynWann::StateE>& e);

double maxval(std::vector<FeynWann::StateE>& e, int bStart, int bStop);
double minval(std::vector<FeynWann::StateE>& e, int bStart, int bStop);

// occupation-related functions
double find_mu(double ncarrier, double t, double mu0, std::vector<double>& E, int nk, int bStart, int bCBM, int bStop);
double find_mu(double ncarrier, double t, double mu0, std::vector<FeynWann::StateE>& e, int bStart, int bCBM, int bStop);
double find_mu(double ncarrier, double t, double mu0, std::vector<diagMatrix>& Ek, int nv);
double compute_ncarrier(bool isHole, double t, double mu, std::vector<diagMatrix>& Ek, int nv);
double compute_ncarrier(bool isHole, double t, double mu, std::vector<FeynWann::StateE>& e, int bStart, int bCBM, int bStop);
std::vector<diagMatrix> computeF(double t, double mu, std::vector<FeynWann::StateE>& e, int bStart, int bStop);

template <typename T> void average_dfde(std::vector<diagMatrix>& F, std::vector<std::vector<T>>& arr, T& avg){
	double sum = 0;
	T Ttmp; avg = 0 * Ttmp; // avg must be initialized to be zero
	for (size_t ik = 0; ik < F.size(); ik++){
		for (int b = 0; b < F[0].size(); b++){
			double dfde = F[ik][b] * (1 - F[ik][b]);
			sum += dfde;
			avg += dfde * arr[ik][b];
		}
	}
	avg = avg / sum;
}