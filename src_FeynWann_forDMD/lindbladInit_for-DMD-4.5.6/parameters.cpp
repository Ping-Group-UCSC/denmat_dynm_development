#include "parameters.h"

static const double omegabyTCut_default = 1e-3;
static const double nEphDelta_default = 4.*sqrt(2); //number of ePhDelta to include in output
static const double degthr_default = 1e-8;

void parameters::read_params(InputMap& inputMap){
	//--- kpoints
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	//--- doping / temperature
	dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional: lowest shift in fermi level from neutral value / VBM in eV (default: 0)
	dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional: highest shift in fermi level from neutral value / VBM in eV (default: 0)
	Tmax = inputMap.get("Tmax") * Kelvin; //maximum temperature in Kelvin (ambient phonon T = initial electron T)
	nkBT = inputMap.get("nkBT", 7); //energy conservation width for e-ph coupling
	carrier_density = inputMap.get("carrier_density", 0); // unit can be 1, cm-1, cm-2 or cm-3 and is undetermined until dimension of the system has been known
	//--- pump
	pumpOmegaMax = inputMap.get("pumpOmegaMax") * eV; //maximum pump frequency in eV
	pumpTau = inputMap.get("pumpTau") * fs; //maximum pump frequency in eV
	probeOmegaMax = inputMap.get("probeOmegaMax") * eV; //maximum probe frequency in eV
	//---select k points, energies, bands and k pairs
	band_skipped = inputMap.get("band_skipped", 0);
	assumeMetal = inputMap.get("assumeMetal", 0);
	assumeMetal_scatt = inputMap.get("assumeMetal_scatt", assumeMetal); if (assumeMetal) assert(assumeMetal_scatt);
	useFinek_for_ERange = inputMap.get("useFinek_for_ERange", 0);
	select_k_use_meff = inputMap.get("select_k_use_meff", 0);
	meff = inputMap.get("meff", 1);
	EBot_set = inputMap.get("EBot_set", 0.) * eV;
	ETop_set = inputMap.get("ETop_set", -1) * eV;
	kparis_eph_eimp = inputMap.get("kparis_eph_eimp", 0);
	read_kpts = inputMap.get("read_kpts", 0); if (read_kpts) { assert(exists("ldbd_data/ldbd_kvec.bin") || exists("ldbd_data/ldbd_kvec_morek.bin")); }
	read_kpts_2nd = inputMap.get("read_kpts_2nd", 0); if (read_kpts_2nd) { assert(read_kpts); assert(exists("ldbd_data/ldbd_kvec.bin")); }
	read_kpairs = inputMap.get("read_kpairs", 0);
	read_erange_brange = inputMap.get("read_erange_brange", 0);
	//--- e-ph control
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	ePhEnabled = (ePhMode != "Off");
	ePhOnlyElec = inputMap.get("ePhOnlyElec", 0); if (assumeMetal) assert(!ePhOnlyElec);
	ePhOnlyHole = inputMap.get("ePhOnlyHole", 0); if (assumeMetal) assert(!ePhOnlyHole);
	bool notBothTrue = !(ePhOnlyElec && ePhOnlyHole); assert(notBothTrue);
	onlyInterValley = inputMap.get("onlyInterValley", 0);
	onlyIntraValley = inputMap.get("onlyIntraValley", 0);
	assert(!onlyInterValley || !onlyIntraValley);
	modeStart = inputMap.get("modeStart", 0);
	modeStop = inputMap.get("modeStop", -1);
	modeSkipStart = inputMap.get("modeSkipStart", 0);
	modeSkipStop = inputMap.get("modeSkipStop", -1);
	//---e-ph paramters
	detailBalance = inputMap.get("detailBalance", 0);
	ePhDelta = inputMap.get("ePhDelta", 0.01) * eV; //energy conservation width for e-ph coupling
	nEphDelta = inputMap.get("nEphDelta", nEphDelta_default); //energy conservation width for e-ph coupling
	omegabyTCut = inputMap.get("omegabyTCut", omegabyTCut_default);
	ePhOmegaCut = inputMap.get("ePhOmegaCut", 1e-6);
	const size_t maxNeighbors = inputMap.get("maxNeighbors", 0); //if non-zero: limit neighbors per k by stochastic down-sampling and amplifying the Econserve weights
	degthr = inputMap.get("degthr", degthr_default);
	// e-imp control and parameters
	defectName = inputMap.getString("defectName");
	iDefect = inputMap.get("iDefect", 1);
	eScattOnlyElec = inputMap.get("eScattOnlyElec", ePhOnlyElec); if (assumeMetal) assert(!eScattOnlyElec);
	eScattOnlyHole = inputMap.get("eScattOnlyHole", ePhOnlyHole); if (assumeMetal) assert(!eScattOnlyHole);
	if (ePhOnlyElec || eScattOnlyElec) eScattOnlyElec = ePhOnlyElec = true;
	if (ePhOnlyHole || eScattOnlyHole) eScattOnlyHole = ePhOnlyHole = true;
	notBothTrue = !(eScattOnlyElec && eScattOnlyHole); assert(notBothTrue);
	detailBalance_defect = inputMap.get("detailBalance_defect", detailBalance);
	scattDelta = inputMap.get("scattDelta", ePhDelta/eV) * eV;
	nScattDelta = inputMap.get("nScattDelta", nEphDelta);
	double omegaLbyT = inputMap.get("omegaLbyT", 1e-4); omegaL = omegaLbyT * Tmax;
	defect_density = inputMap.get("defect_density", 0);
	//---output control
	DEBUG = inputMap.get("DEBUG", 0);
	needConventional = inputMap.get("needConventional", 0);
	writeU = inputMap.get("writeU", 0); // for electron-impurity and electron-electron scattering models
	write_sparseP = inputMap.get("write_sparseP", 0);
	writegm = inputMap.get("writegm", true);
	keepgm = inputMap.get("keepgm", false); if (keepgm) assert(writegm);// if (keepgm) assert(modeSkipStop <= modeSkipStart);
	mergegm = inputMap.get("mergegm", false); if (mergegm) assert(writegm);// if (mergegm) assert(modeSkipStop <= modeSkipStart);
	layerOcc = inputMap.get("layerOcc", 0);
	writeHEz = inputMap.get("writeHEz", 0);
	needOmegaPhMax = inputMap.get("needOmegaPhMax", ePhEnabled);
	save_dHePhSum_disk = inputMap.get("save_dHePhSum_disk", ePhEnabled); if (!ePhEnabled) assert(!save_dHePhSum_disk); // save dHePhSum in disk to save memory
	//---gfactor
	bool orbitalZeeman = inputMap.getBool("orbitalZeeman", false);
	needL = inputMap.get("needL", orbitalZeeman);
	if (!needL){
		gfac_mean = inputMap.getVector("gfac_mean", vector3<>(2.0023193043625635, 2.0023193043625635, 2.0023193043625635));
		gfac_sigma = inputMap.getVector("gfac_sigma", vector3<>(0, 0, 0));
		gfac_cap = inputMap.getVector("gfac_cap", vector3<>(0, 0, 0));
		read_gfack = inputMap.get("read_gfack", 0); if (read_gfack) assert(read_kpts);
		if (read_gfack) assert(fileSize("g_tensor_k.dat") > 0);
	}
	else{
		gfac_mean = vector3<>(2.0023193043625635, 2.0023193043625635, 2.0023193043625635);
		gfac_sigma = vector3<>(0, 0, 0);
		gfac_cap = vector3<>(0, 0, 0);
		read_gfack = false;
	}
	//---matrix element analysis
	analyse_g2_E1E2fix = inputMap.get("analyse_g2_E1E2fix", 0); if (analyse_g2_E1E2fix) assert(DEBUG);
	E1fix = inputMap.get("E1fix", 0) / 1000. *eV; // reference to CBM(VBM) if >(<) 0
	E2fix = inputMap.get("E2fix", 0) / 1000. *eV; // reference to CBM(VBM) if >(<) 0

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("nkBT = %lg\n", nkBT);

	logPrintf("pumpOmegaMax = %lg\n", pumpOmegaMax);
	logPrintf("pumpTau = %lg\n", pumpTau);
	logPrintf("probeOmegaMax = %lg\n", probeOmegaMax);

	logPrintf("band_skipped = %d\n", band_skipped);
	logPrintf("assumeMetal = %d\n", assumeMetal);
	logPrintf("assumeMetal_scatt = %d\n", assumeMetal_scatt);
	logPrintf("useFinek_for_ERange = %d\n", useFinek_for_ERange);
	logPrintf("select_k_use_meff = %d\n", select_k_use_meff);
	logPrintf("meff = %lg\n", meff);
	logPrintf("EBot_set = %lg\n", EBot_set);
	logPrintf("ETop_set = %lg\n", ETop_set);
	logPrintf("kparis_eph_eimp = %d\n", kparis_eph_eimp);
	logPrintf("read_kpts = %d\n", read_kpts);
	logPrintf("read_kpts_2nd = %d\n", read_kpts_2nd);
	logPrintf("read_kpairs = %d\n", read_kpairs);
	logPrintf("read_erange_brange = %d\n", read_kpairs);

	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("ePhOnlyElec = %d\n", ePhOnlyElec);
	logPrintf("ePhOnlyHole = %d\n", ePhOnlyHole);
	logPrintf("onlyInterValley = %d\n", onlyInterValley);
	logPrintf("onlyIntraValley = %d\n", onlyIntraValley);
	logPrintf("modeStart = %d\n", modeStart);
	logPrintf("modeStop = %d\n", modeStop);
	logPrintf("modeSkipStart = %d\n", modeSkipStart);
	logPrintf("modeSkipStop = %d\n", modeSkipStop);

	logPrintf("detailBalance = %d\n", detailBalance);
	logPrintf("ePhDelta = %lg\n", ePhDelta);
	logPrintf("nEphDelta = %lg\n", nEphDelta);
	logPrintf("omegabyTCut = %lg\n", omegabyTCut);
	logPrintf("ePhOmegaCut = %lg\n", ePhOmegaCut);
	logPrintf("maxNeighbors = %lu\n", maxNeighbors);
	logPrintf("degthr = %lg\n", degthr);

	logPrintf("defectName = %s\n", defectName.c_str());
	logPrintf("iDefect = %d\n", iDefect);
	logPrintf("eScattOnlyElec = %d\n", eScattOnlyElec);
	logPrintf("eScattOnlyHole = %d\n", eScattOnlyHole);
	logPrintf("scattDelta = %lg\n", scattDelta);
	logPrintf("nScattDelta = %lg\n", nScattDelta);
	logPrintf("detailBalance_defect = %d\n", detailBalance_defect);
	logPrintf("omegaL = %lg\n", omegaL);

	logPrintf("DEBUG = %d\n", DEBUG);
	logPrintf("needConventional = %d\n", needConventional);
	logPrintf("writeU = %d\n", writeU);
	logPrintf("write_sparseP = %d\n", write_sparseP);
	logPrintf("writegm = %d\n", writegm);
	logPrintf("keepgm = %d\n", keepgm);
	logPrintf("mergegm = %d\n", mergegm);
	logPrintf("layerOcc = %d\n", layerOcc);
	logPrintf("writeHEz = %d\n", writeHEz);
	logPrintf("needOmegaPhMax = %d\n", needOmegaPhMax);
	logPrintf("save_dHePhSum_disk = %d\n", save_dHePhSum_disk);

	logPrintf("needL = %d\n", needL);
	logPrintf("gfac_mean = %lg %lg %lg\n", gfac_mean[0], gfac_mean[1], gfac_mean[2]);
	logPrintf("gfac_sigma = %lg %lg %lg\n", gfac_sigma[0], gfac_sigma[1], gfac_sigma[2]);
	logPrintf("gfac_cap = %lg %lg %lg\n", gfac_cap[0], gfac_cap[1], gfac_cap[2]);
	logPrintf("read_gfack = %d\n", read_gfack);

	if (analyse_g2_E1E2fix){
		logPrintf("analyse_g2_E1E2fix = %d\n", analyse_g2_E1E2fix);
		logPrintf("E1fix = %lg\n", E1fix);
		logPrintf("E2fix = %lg\n", E2fix);
	}
}