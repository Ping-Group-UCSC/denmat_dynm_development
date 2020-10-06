#pragma once
#include "common_headers.h"
#include "parameters.h"

struct elecelecParam{
	string eeMode;
	bool antisymmetry;
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
	// there must be only one impurity level
	string impMode;
	bool partial_ionized;
	double ni, ni_cell, Z, g, lng, ni_ionized, Eimp; // ni > 0 - n-dope; ni < 0 - p-dope; Z must > 0
	                                                 // ni_cell = impurity_density * nk_full * latt->volume
	double carrier_cell_gs, ne_cell_gs, nh_cell_gs; // carrier_cell = carrier_density * nk_full * latt->volume; gs - ground state

	double carrier_cell_ex(bool isHole, double n_cell_ex){
		if (ni == 0 || (ni > 0 && isHole) || (ni < 0 && !isHole)) return n_cell_ex;
		else if (ni < 0 && isHole) return carrier_cell_gs + (n_cell_ex - nh_cell_gs);
		else if (ni > 0 && !isHole) return carrier_cell_gs + (n_cell_ex - ne_cell_gs);
	}
	
	void calc_ni_ionized(double t, double mu, bool silent = true){
		ni_ionized = ni * (1 - occ_of_impurity_level(ni < 0, t, mu));
		if (!silent && ionode && ni != 0) printf("ni_ionized = %lg ratio = %lg\n", ni_ionized, ni_ionized / ni);
	}
	void calc_ni_ionized(bool isHole, double t, double mu, bool silent = true){
		if (isHole && ni < 0 || !isHole && ni > 0){
			ni_ionized = ni * (1 - occ_of_impurity_level(isHole, t, mu));
			if (!silent && ionode) printf("ni_ionized = %lg ratio = %lg\n", ni_ionized, ni_ionized / ni);
		}
	}
	double compute_carrier_cell_of_impurity_level(bool isHole, double t, double mu){
		return Z * ni_cell * occ_of_impurity_level(isHole, t, mu);
	}
	double occ_of_impurity_level(bool isHole, double t, double mu){
		if (ni == 0 || !partial_ionized || (isHole && ni > 0) || (!isHole && ni < 0)) return 0;
		double ebyt = (ni > 0 ? (Eimp - mu) : (mu - Eimp)) / t - lng;
		if (ebyt < -46) return 1;
		else if (ebyt > 46) return 0;
		else return 1. / (exp(ebyt) + 1);
	}
};

extern coulombParam clp;
extern elecimpParam eip;
extern elecelecParam eep;