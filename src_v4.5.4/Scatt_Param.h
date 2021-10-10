#pragma once
#include "common_headers.h"
#include "parameters.h"

struct elecelecParam{
	string eeMode;
	bool antisymmetry;
	double degauss;
};

struct coulombParam{
	string scrMode; // "none", "medium"
	string scrFormula; // "debye", "lindhard", "RPA"
	string dynamic;
	bool update, ovlp, fderavitive_technique; // if ovlp, there is (n,n') sum with overlap, otherwise, (n) sum
	double eppa, meff; // user-defined plasmon-pole energy, if zero, plasma frequency
	int nomega; double omegamax; // real-energy axis, not implemented
	double nfreetot, eps, smearing;

	void check_params(){
		if (scrMode != "none" && scrMode != "medium")
			error_message("scrMode must be none or medium now","coulombParam");
		if (scrFormula != "debye" && scrFormula != "Bechstedt" && scrFormula != "lindhard" && scrFormula != "RPA")
			error_message("scrFormula must be debye, Bechstedt, lindhard or RPA now","coulombParam");
		if (dynamic != "static" && (scrFormula == "debye" || scrFormula == "Bechstedt"))
			error_message("all models are inn static limit");
		if (dynamic != "static" && dynamic != "ppa")
			error_message("dynamic must be static or ppa now", "coulombParam");
		if (eppa < 0)
			error_message("eppa must >= 0", "coulombParam");
		if (meff <= 0)
			error_message("meff must > 0", "coulombParam");
		if (dynamic == "real-axis" && nomega < 2)
			error_message("real-axis at least needs 2 frequencies","coulombParam");
		if (dynamic == "real-axis" && omegamax < 0)
			error_message("omegamax < 0 is not allowed","coulombParam");
		if (!fderavitive_technique && smearing == 0)
			error_message("for static limit, you need either fderavitive_technique or smearing > 0", "coulombParam");
		if (smearing < 0)
			error_message("smearing must >= 0","coulombParam");
		if (eps < 1)
			error_message("epsilon_background < 1 is not right","coulombParam");
	}
};

struct elecimpParam{
	std::vector<string> impMode;
	std::vector<double> ni, ni_bvk, ni_ionized, Z; // ni > 0 - n-dope; ni < 0 - p-dope; Z must > 0
                                                   // ni_bvk = impurity_density * nk_full * latt->volume
	std::vector<bool> partial_ionized;
	std::vector<double> Eimp, g, lng;
	double carrier_bvk_gs, ne_bvk_gs, nh_bvk_gs; // carrier_bvk = carrier_density * nk_full * latt->volume; gs - ground state
	std::vector<double> degauss;

	double carrier_bvk_ex(bool isHole, double n_bvk_ex){
		if (ni.size() == 0 || (carrier_bvk_gs > 0 && isHole) || (carrier_bvk_gs < 0 && !isHole)) return n_bvk_ex;
		else if (carrier_bvk_gs < 0 && isHole) return carrier_bvk_gs + (n_bvk_ex - nh_bvk_gs);
		else if (carrier_bvk_gs > 0 && !isHole) return carrier_bvk_gs + (n_bvk_ex - ne_bvk_gs);
	}
	
	void calc_ni_ionized(double t, double mu, bool silent = true){
		for (int iD = 0; iD < ni.size(); iD++){
			ni_ionized[iD] = ni[iD] * (1 - occ_of_impurity_level(iD, ni[iD] < 0, t, mu));
			if (!silent && ionode) printf("iD = %d ni_ionized = %lg ratio = %lg\n", iD, ni_ionized[iD], ni_ionized[iD] / ni[iD]);
		}
	}
	void calc_ni_ionized(bool isHole, double t, double mu, bool silent = true){
		for (int iD = 0; iD < ni.size(); iD++){
			if (isHole && ni[iD] > 0 || !isHole && ni[iD] < 0) continue;
			ni_ionized[iD] = ni[iD] * (1 - occ_of_impurity_level(iD, isHole, t, mu));
			if (!silent && ionode) printf("iD = %d ni_ionized = %lg ratio = %lg\n", iD, ni_ionized[iD], ni_ionized[iD] / ni[iD]);
		}
	}
	double compute_carrier_bvk_of_impurity_level(bool isHole, double t, double mu){
		double result = 0;
		for (int iD = 0; iD < ni.size(); iD++)
			result += Z[iD] * ni_bvk[iD] * occ_of_impurity_level(iD, isHole, t, mu);
		return result;
	}
	double occ_of_impurity_level(int iD, bool isHole, double t, double mu){
		if (ni[iD] == 0 || !partial_ionized[iD] || (isHole && ni[iD] > 0) || (!isHole && ni[iD] < 0)) return 0;
		double ebyt = (ni[iD] > 0 ? (Eimp[iD] - mu) : (mu - Eimp[iD])) / t - lng[iD];
		if (ebyt < -46) return 1;
		else if (ebyt > 46) return 0;
		else return 1. / (exp(ebyt) + 1);
	}
};

extern coulombParam clp;
extern elecimpParam eip;
extern elecelecParam eep;