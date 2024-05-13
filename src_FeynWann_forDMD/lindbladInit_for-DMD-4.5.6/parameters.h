#pragma once
#include "common_headers.h"

class parameters{
public:
	vector3<int> NkMult;

	double dmuMin, dmuMax, dmu, carrier_density, Tmax, nkBT;
	bool needL, read_gfack;
	vector3<> gfac_mean, gfac_sigma, gfac_cap;
	double pumpOmegaMax, pumpTau, probeOmegaMax;

	int band_skipped; bool assumeMetal, assumeMetal_scatt, useFinek_for_ERange, select_k_use_meff;
	double EBot_set, ETop_set, meff;
	bool read_erange_brange, read_kpts, read_kpairs, kparis_eph_eimp, read_kpts_2nd;

	bool ePhEnabled, onlyInterValley, onlyIntraValley, ePhOnlyElec, ePhOnlyHole, eScattOnlyElec, eScattOnlyHole; //!< whether e-ph coupling is enabled
	int modeStart, modeStop, modeSkipStart, modeSkipStop;
	string defectName; int iDefect; double defect_density;

	bool detailBalance, detailBalance_defect;
	double ePhDelta, scattDelta, nEphDelta, nScattDelta, omegabyTCut, degthr, omegaL, ePhOmegaCut; //!< Gaussian energy conservation width

	bool needConventional, write_sparseP, writeU, needOmegaPhMax, layerOcc, writeHEz, writegm, keepgm, mergegm, save_dHePhSum_disk;

	// matrix element analysis
	bool analyse_g2_E1E2fix;
	double E1fix, E2fix;

	parameters(){}

	void read_params(InputMap& inputMap);
};