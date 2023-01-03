#include "parameters.h"
#include "mymp.h"
#include "PumpProbe.h"
#include "Scatt_Param.h"
#include "ODE.h"

void parameters::read_jdftx(){
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	for (int i = 0; i < 5; i++)
		fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		sscanf(s, "%le", &temperature); if (ionode) printf("temperature = %14.7le\n", temperature);
	}
	if (fgets(s, sizeof s, fp) != NULL){
		double dtmp1, dtmp2;
		sscanf(s, "%lg %lg %lg", &dtmp1, &dtmp2, &mu); if (ionode) printf("mu = %lg\n", mu);
	}
	if (fgets(s, sizeof s, fp) != NULL){
		sscanf(s, "%lg %lg", &degauss, &ndegauss); if (ionode) printf("degauss = %lg ndegauss = %lg\n", degauss, ndegauss);
	}
	fclose(fp);
}

void parameters::get_valley_transitions(string file_forbid_vtrans){
	vtrans.resize(vpos.size());
	for (int iv = 0; iv < vpos.size(); iv++)
		vtrans[iv].resize(vpos.size(), true);
	if (file_forbid_vtrans != "NONE"){
		FILE* fp = fopen(file_forbid_vtrans.c_str(), "r");
		char s[10];
		while (fgets(s, sizeof s, fp) != NULL){
			int i, j;
			sscanf(s, "%d %d", &i, &j); if (ionode) printf("forbidden valley transition: %d %d\n", i, j);
			vtrans[i][j] = false; vtrans[j][i] = false;
		}
		fclose(fp);
	}
	if (ionode){
		printf("\nAllowed transitions:\n");
		for (int iv1 = 0; iv1 < vpos.size(); iv1++)
		for (int iv2 = 0; iv2 < vpos.size(); iv2++)
			if (vtrans[iv1][iv2]) printf("%d %d\n", iv1, iv2);
		printf("\n");
	}
}

void parameters::read_param(){
	fstream fin;
	fin.open("param.in", ios::in);
	if (fin.fail()) error_message("input file param.in does not exist");
	std::map<std::string, std::string> param_map = map_input(fin);

	if (ionode) printf("\n");
	if (ionode) printf("==================================================\n");
	if (ionode) printf("==================================================\n");
	if (ionode) printf("reading input parameters\n");
	if (ionode) printf("==================================================\n");
	if (ionode) printf("==================================================\n");

	if (ionode) printf("\n**************************************************\n");
	if (ionode) printf("Code control parameters:\n");
	if (ionode) printf("**************************************************\n");
	DEBUG = get(param_map, "DEBUG", false);
	if (ionode && DEBUG && !is_dir("debug_info")) system("mkdir debug_info");
	restart = get(param_map, "restart", false);
	if (ionode && !restart && is_dir("restart"))
		error_message("diretory restart presents, you should run a restart calculation");
	code = getString(param_map, "code", "jdftx");
	material_model = getString(param_map, "material_model", "none");
	if (ionode && !restart && material_model == "none") system("mkdir restart");
	MPI_Barrier(MPI_COMM_WORLD);
	compute_tau_only = get(param_map, "compute_tau_only", false);

	if (ionode) printf("\n**************************************************\n");
	if (ionode) printf("Master equation control parameters:\n");
	if (ionode) printf("**************************************************\n");
	alg.scatt_enable = get(param_map, "alg_scatt_enable", true);
	alg.eph_enable = get(param_map, "alg_eph_enable", true);
	modeStart = get(param_map, "modeStart", 0);
	modeEnd = get(param_map, "modeEnd", -1);
	scale_scatt = get(param_map, "scale_scatt", 1.);
	scale_eph = get(param_map, "scale_eph", scale_scatt);
	scale_ei = get(param_map, "scale_ei", scale_scatt);
	scale_ee = get(param_map, "scale_ee", scale_scatt);
	alg.only_eimp = get(param_map, "alg_only_eimp", false);
	alg.only_ee = get(param_map, "alg_only_ee", false);
	alg.only_intravalley = get(param_map, "alg_only_intravalley", false);
	alg.only_intervalley = get(param_map, "alg_only_intervalley", false);
	// for eph, if !need_elec and !need_hole, electrons and holes are treated together
	// in model cases, usually alg_eph_sepr_eh is implied true and (alg_eph_need_elec xor alg_eph_need_hole)
	alg.eph_sepr_eh = get(param_map, "alg_eph_sepr_eh", true);
	if (!alg.eph_sepr_eh){
		alg.eph_need_elec = true; alg.eph_need_hole = true;
	}
	else{
		alg.eph_need_elec = get(param_map, "alg_eph_need_elec", true);
		alg.eph_need_hole = get(param_map, "alg_eph_need_hole", false);
	}

	alg.semiclassical = get(param_map, "alg_semiclassical", 0);
	alg.summode = get(param_map, "alg_summode", 1);
	alg.ddmdteq = get(param_map, "alg_ddmdteq", 0);
	alg.expt = get(param_map, "alg_expt", 1);
	alg.expt_elight = get(param_map, "alg_expt_elight", alg.expt);
	alg.scatt = getString(param_map, "alg_scatt", "lindblad");
	alg.picture = getString(param_map, "alg_picture", "interaction");
	if (alg.picture == "non-interaction") alg.picture = "schrodinger";
	alg.linearize = get(param_map, "alg_linearize", false);
	alg.linearize_dPee = get(param_map, "alg_linearize_dPee", false);
	need_imsig = get(param_map, "need_imsig", !alg.linearize);

	if (ionode) printf("\nmodel intermal magnetic field parameters:\n");
	alg.modelH0hasBS = get(param_map, "alg_modelH0hasBS", 1);
	alg.read_Bso = get(param_map, "alg_read_Bso", 0);

	if (ionode) printf("\nsparse parameters:\n");
	alg.Pin_is_sparse = get(param_map, "alg_Pin_is_sparse", 0);
	alg.sparseP = get(param_map, "alg_sparseP", 0);
	alg.thr_sparseP = get(param_map, "alg_thr_sparseP", 1e-40);

	if (ionode) printf("\nphenomenological relaxation parameters:\n");
	alg.phenom_relax = get(param_map, "alg_phenom_relax", 0);
	if (alg.phenom_relax){
		tau_phenom = get(param_map, "tau_phenom", 1e15, ps);
		bStart_tau = get(param_map, "bStart_tau", 0); // relative to bStart_dm
		bEnd_tau = get(param_map, "bEnd_tau", 0);
	}

	if (ionode) printf("\nSmearing parameters:\n");
	double degauss_input = 0, ndegauss_input = 0;
	double mu_input = 0;
	if (material_model == "none"){
		read_jdftx();
		degauss_input = degauss;
		ndegauss_input = ndegauss;
		mu_input = mu;
	}
	degauss = get(param_map, "degauss", degauss_input / eV, eV);
	ndegauss = get(param_map, "ndegauss", ndegauss_input);

	if (ionode) printf("\n**************************************************\n");
	if (ionode) printf("System and external condition parameters:\n");
	if (ionode) printf("**************************************************\n");
	if (material_model != "none"){
		temperature = get(param_map, "temperature", 300, Kelvin);

		nk1 = get(param_map, "nk1", 1);
		nk2 = get(param_map, "nk2", 1);
		nk3 = get(param_map, "nk3", 1);
		ewind = get(param_map, "ewind", 6) * temperature;
		lattvec1 = getVector(param_map, "lattvec1", vector3<>(1., 0., 0.));
		lattvec2 = getVector(param_map, "lattvec2", vector3<>(0., 1., 0.));
		lattvec3 = getVector(param_map, "lattvec3", vector3<>(0., 0., 1.));
		matrix3<> Rtmp(lattvec1[0], lattvec2[0], lattvec3[0],
			lattvec1[1], lattvec2[1], lattvec3[1],
			lattvec1[2], lattvec2[2], lattvec3[2]);
		R = Rtmp;
	}
	int dim_default = 3;
	if (FILE *fp = fopen("ldbd_data/ldbd_R.dat", "r")){
		fscanf(fp, "%d", &dim_default); fclose(fp);
	}
	dim = get(param_map, "dim", dim_default);
	if (dim == 2) thickness = get(param_map, "thickness", 0);
	scissor = get(param_map, "scissor", 0, eV);

	mu = get(param_map, "mu", mu_input / eV, eV);
	carrier_density = get(param_map, "carrier_density", 0, std::pow(bohr2cm, dim));
	carrier_density_means_excess_density = get(param_map, "carrier_density_means_excess_density", 0);

	// magnetic field
	if (ionode) printf("\nmagnetic field parameters:\n");
	Bx = get(param_map, "Bx", 0., Tesla2au);
	By = get(param_map, "By", 0., Tesla2au);
	Bz = get(param_map, "Bz", 0., Tesla2au);
	B[0] = Bx; B[1] = By; B[2] = Bz;
	// g factor
	gfac_normal_dist = get(param_map, "gfac_normal_dist", 0);
	gfac_k_resolved = get(param_map, "gfac_k_resolved", 1);
	gfac_mean = get(param_map, "gfac_mean", 2);
	gfac_sigma = get(param_map, "gfac_sigma", 0);
	gfac_cap = get(param_map, "gfac_cap", 0);
	double gfac_mean, gfac_sigma, gfac_cap;
	// electric field
	scale_Ez = get(param_map, "scale_Ez", 0.); // scaling factor of HEz

	if (ionode) printf("\n**************************************************\n");
	if (ionode) printf("Screening and model e-i and e-e parameters:\n");
	if (ionode) printf("**************************************************\n");
	if (ionode) printf("\nscreening parameters:\n");
	clp.scrMode = getString(param_map, "scrMode", "none");
	clp.scrFormula = getString(param_map, "scrFormula", "RPA");
	//clp.ovlp = clp.scrFormula == "RPA" ? true : false;
	clp.update = get(param_map, "update_screening", 1);
	clp.dynamic = getString(param_map, "dynamic_screening", "static");
	clp.ppamodel = getString(param_map, "ppamodel", "gn");
	clp.eppa = get(param_map, "eppa_screening", 0);
	clp.meff = get(param_map, "meff_screening", 1);
	clp.omegamax = get(param_map, "omegamax_screening", 0, eV);
	clp.nomega = get(param_map, "nomega_screening", 1);
	if (clp.dynamic == "ppa" && clp.ppamodel == "gn") clp.nomega = 2;
	clp.dynamic_screening_ee_two_freqs = get(param_map, "dynamic_screening_ee_two_freqs", 0);
	clp.fderavitive_technique = get(param_map, "fderavitive_technique_static_screening", 1);
	clp.smearing = get(param_map, "smearing_screening", 0, eV);
	clp.eps = get(param_map, "epsilon_background", 1);

	// to turn on electron-impurity scattering, set eip.ni nonzero
	if (ionode) printf("\nelectron-impurity parameters:\n");
	while (true){
		string dfName = eip.ni.size() == 0 ? "" : int2str(int(eip.ni.size()) + 1);
		double nitmp = get(param_map, "impurity_density" + dfName, 0, std::pow(bohr2cm, dim));
		if (nitmp == 0) break;
		eip.ni.push_back(nitmp);
		eip.impMode.push_back(getString(param_map, "impMode" + dfName, "model_ionized"));
		eip.partial_ionized.push_back(get(param_map, "partial_ionized" + dfName, 0));
		eip.Z.push_back(get(param_map, "Z_impurity" + dfName, 1));
		eip.g.push_back(get(param_map, "g_impurity" + dfName, 2));
		//eip.lng.push_back(std::log(eip.g[ni.size()-1])); if (ionode) printf("lng = %lg\n", eip.lng[ni.size()-1]);
		eip.Eimp.push_back(get(param_map, "E_impurity" + dfName, mu, eV));
		eip.degauss.push_back(get(param_map, "degauss_eimp", degauss / eV, eV));
		eip.detailBalance.push_back(get(param_map, "detailBalance", 0));
	}
	eip.ni_bvk.resize(eip.ni.size()); eip.ni_ionized.resize(eip.ni.size());
	freq_update_eimp_model = get(param_map, "freq_update_eimp_model", 0);

	if (ionode) printf("\nelectron-electron parameters:\n");
	eep.eeMode = getString(param_map, "eeMode", "none"); // "none" will turn off electron-electron scattering
	eep.antisymmetry = get(param_map, "ee_antisymmetry", 0);
	eep.degauss = get(param_map, "degauss_ee", degauss / eV, eV);
	freq_update_ee_model = get(param_map, "freq_update_ee_model", 0);

	if (ionode) printf("\n**************************************************\n");
	if (ionode) printf("Spin generation and measurement parameters:\n");
	if (ionode) printf("**************************************************\n");
	// laser and probe
	if (ionode) printf("\npump and probe parameters:\n");
	pmp.laserMode = getString(param_map, "laserMode", "pump");
	string pumpMode = getString(param_map, "pumpMode", "perturb"); //to be consistent with previous version
	pmp.laserAlg = getString(param_map, "laserAlg", pumpMode);
	double pumpA0 = get(param_map, "pumpA0", 0.); //to be consistent with previous version
	pmp.laserA = get(param_map, "laserA", pumpA0);
	double pumpE = get(param_map, "pumpE", 0., eV); //to be consistent with previous version
	pmp.laserE = get(param_map, "laserE", pumpE / eV, eV);
	pmp.pumpTau = get(param_map, "pumpTau", 0., fs);
	if (!restart){
		if (material_model == "none"){
			pmp.pump_tcenter = get(param_map, "pump_tcenter", (t0 + 5 * pmp.pumpTau) / fs, fs); // 5*Tau is quite enough
			FILE *filtime = fopen("restart/pump_tcenter.dat", "w"); fprintf(filtime, "%14.7le", pmp.pump_tcenter); fclose(filtime);
		}
	}
	else{
		if (FILE *ftime = fopen("restart/pump_tcenter.dat", "r")){
			char s[200];
			if (fgets(s, sizeof s, ftime) != NULL){
				sscanf(s, "%le", &pmp.pump_tcenter); if (ionode) printf("pmp.pump_tcenter = %lg\n", pmp.pump_tcenter);
			}
			fclose(ftime);
		}
		else
			error_message("restart needs restart/pump_tcenter.dat");
	}
	if (pmp.laserA > 0){
		string pumpPoltype = getString(param_map, "pumpPoltype", "NONE");
		pmp.laserPoltype = getString(param_map, "laserPoltype", pumpPoltype);
		pmp.laserPol = pmp.set_Pol(pmp.laserPoltype);
		if (ionode) { pmp.print(pmp.laserPol); }
		while (true){
			int iPol = int(pmp.probePol.size()) + 1;
			ostringstream oss; oss << iPol;
			string polName = oss.str();
			string poltype = getString(param_map, "probePoltype" + polName, "NONE");
			if (poltype == "NONE") break;
			vector3<complex> pol = pmp.set_Pol(poltype);
			pmp.probePoltype.push_back(poltype);
			pmp.probePol.push_back(pol);
			if (ionode) { pmp.print(pmp.probePol[iPol - 1]); }
		}
		if (pmp.probePol.size() > 0){
			pmp.probeEmin = get(param_map, "probeEmin", 0., eV);
			pmp.probeEmax = get(param_map, "probeEmax", 0., eV);
			pmp.probeDE = get(param_map, "probeDE", 0., eV);
			pmp.probeNE = int(ceil((pmp.probeEmax - pmp.probeEmin) / pmp.probeDE + 1e-6));
			if (ionode) printf("probeNE = %d\n", pmp.probeNE);
			pmp.probeTau = get(param_map, "probeTau", 0., fs);
		}
	}

	// magnetic field perturbation
	if (ionode) printf("\nmagnetic field perturbation parameters:\n");
	Bxpert = get(param_map, "Bxpert", 0., Tesla2au);
	Bypert = get(param_map, "Bypert", 0., Tesla2au);
	Bzpert = get(param_map, "Bzpert", 0., Tesla2au);
	Bpert[0] = Bxpert; Bpert[1] = Bypert; Bpert[2] = Bzpert;
	needL = get(param_map, "needL", 0);

	// time paramters and studying system parameters
	if (ionode) printf("\nTime control parameters:\n");
	if (!restart)
		t0 = get(param_map, "t0", 0., fs);
	else{
		if (FILE *ftime = fopen("restart/time_restart.dat", "r")){
			char s[200];
			if (fgets(s, sizeof s, ftime) != NULL){
				sscanf(s, "%le", &t0); if (ionode) printf("t0 = %lg\n", t0);
			}
			fclose(ftime);
		}
		else
			error_message("restart needs restart/time_restart.dat");
	}
	tend = get(param_map, "tend", 0., fs);
	tstep = get(param_map, "tstep", 1., fs);
	if (pmp.laserAlg == "lindblad" || pmp.laserAlg == "coherent"){
		double tstep_pump = get(param_map, "tstep_pump", tstep / fs, fs);
		tstep_laser = get(param_map, "tstep_laser", tstep_pump / fs, fs);
	}
	else
		tstep_laser = tstep;

	if (ionode) printf("\nOther measurement parameters:\n");
	print_tot_band = get(param_map, "print_tot_band", 0);
	alg.set_scv_zero = get(param_map, "alg_set_scv_zero", 0);
	freq_measure = get(param_map, "freq_measure", 1);
	freq_measure_ene = get(param_map, "freq_measure_ene", 10);
	freq_compute_tau = get(param_map, "freq_compute_tau", freq_measure_ene);
	de_measure = get(param_map, "de_measure", 5e-4, eV);
	degauss_measure = get(param_map, "degauss_measure", 2e-3, eV);

	//valley positions
	while (true){
		int iv = int(vpos.size()) + 1;
		ostringstream oss; oss << iv;
		string vName = oss.str();
		vector3<> v3tmp = getVector(param_map, "valley" + vName, vector3<>(1, 1, 1));
		if (abs(v3tmp[0]) >= 1 || abs(v3tmp[1]) >= 1 || abs(v3tmp[2]) >= 1){
			if (ionode) printf("invalid valley position. all values must be within (-1,1)\n");
			break;
		}
		vpos.push_back(v3tmp);
	}
	if (vpos.size() == 1) error_message("if you want to analyse valley dynamics, at least you need provide positions of two valleys", "read_param");
	if (ionode) printf("number of valleys: %lu\n", vpos.size());
	if (vpos.size() >= 2){
		string file_forbid_vtrans = getString(param_map, "file_forbid_vtrans", "NONE");
		get_valley_transitions(file_forbid_vtrans);
	}
	type_q_ana = getString(param_map, "type_q_ana", "wrap_around_valley");

	rotate_spin_axes = get(param_map, "rotate_spin_axes", 0);
	sdir_z = normalize(getVector(param_map, "sdir_z", vector3<>(0., 0., 1.)));
	sdir_x = normalize(getVector(param_map, "sdir_x", vector3<>(1., 0., 0.)));
	if (fabs(dot(sdir_z, sdir_x)) > 1e-10)
		error_message("sdir_z and sdir_x must be orthogonal", "read_param");
	vector3<> sdir_y = normalize(cross(sdir_z, sdir_x));
	if (ionode) printf("sdir_y: %lg %lg %lg\n", sdir_y[0], sdir_y[1], sdir_y[2]);
	sdir_rot.set_rows(sdir_x, sdir_y, sdir_z);

	if (ionode) printf("\nODE parameters:\n");
	alg.ode_method = getString(param_map, "alg_ode_method", "rkf45");
	// ODE (ordinary derivative equation) parameters
	ode.hstart = get(param_map, "ode_hstart", 1e-3, fs);
	ode.hmin = get(param_map, "ode_hmin", 0, fs);
	ode.hmax = get(param_map, "ode_hmax", std::max(tstep, tstep_laser) / fs, fs);
	double dtmp = pmp.laserAlg == "coherent" ? 1 : tstep_laser / fs;
	ode.hmax_laser = get(param_map, "ode_hmax_laser", dtmp, fs);
	ode.epsabs = get(param_map, "ode_epsabs", 1e-8);

	/*
	if (ionode) printf("\nkpath realted parameters:\n");
	print_along_kpath = get(param_map, "print_along_kpath", false);
	if (ionode && !restart && print_along_kpath){
		if (is_dir("ddm_along_kpath_results")) system("rm -r ddm_along_kpath_results");
		system("mkdir ddm_along_kpath_results");
	}
	while (print_along_kpath){
		int iPath = int(kpath_start.size()) + 1;
		ostringstream oss; oss << iPath;
		string pathName = oss.str();
		vector3<double> kstart = getVector(param_map, "kpath_start" + pathName, vector3<>(2, 2, 2));
		vector3<double> kend = getVector(param_map, "kpath_end" + pathName, vector3<>(2, 2, 2));
		if (kstart == vector3<>(2, 2, 2) || kend == vector3<>(2, 2, 2)) break;
		if (kend == kstart) error_message("kstart == kend");
		kpath_start.push_back(kstart);
		kpath_end.push_back(kend);
	}
	if (print_along_kpath && kpath_start.size() == 0) error_message("print_along_kpath && kpath_start.size() == 0");
	*/

	/*
	if (!needL){
		if (ionode) printf("\nDP mechanism analysis parameters:\n");
		alg.use_dmDP_taufm_as_init = get(param_map, "alg_use_dmDP_taufm_as_init", false);
		alg.DP_beyond_carrierlifetime = get(param_map, "alg_DP_beyond_carrierlifetime", false);
		alg.mix_tauneq = get(param_map, "alg_mix_tauneq", 0.2);
		alg.positive_tauneq = get(param_map, "alg_positive_tauneq", false);
		alg.use_dmDP_in_evolution = get(param_map, "alg_use_dmDP_in_evolution", false);
	}
	*/

	if (ionode) printf("\nother parameters:\n");
	degthr = get(param_map, "degthr", 1e-6);
	band_skipped = get(param_map, "band_skipped", 0);
	if (ionode) printf("\n\n");

	//////////////////////////////////////////////////
	// check availability of input parameters
	//////////////////////////////////////////////////
	if (code != "jdftx")
		error_message("code value is not allowed", "read_param");
	if (code == "jdftx" && !alg.summode)
		error_message("if code is jdftx, alg_summode must be true", "read_param");
	if (material_model == "mos2" || material_model == "gaas"){
		if (!alg.eph_sepr_eh)
			error_message("if material_model is mos2 or gaas, alg_eph_sepr_eh must be true", "read_param");
		if (pmp.laserA > 0)
			error_message("for model, laser is not allowed", "read_param");
	}
	if (alg.DP_beyond_carrierlifetime && restart)
		error_message("alg_DP_beyond_carrierlifetime && param->restart", "read_param");
	if (alg.DP_beyond_carrierlifetime && !compute_tau_only)
		error_message("alg_DP_beyond_carrierlifetime && !compute_tau_only", "read_param");
	if (alg.DP_beyond_carrierlifetime && Bpert.length() < 1e-12)
		error_message("alg_DP_beyond_carrierlifetime && Bpert.length() < 1e-12", "read_param");
	if (alg.DP_beyond_carrierlifetime && alg.picture != "schrodinger")
		error_message("alg_DP_beyond_carrierlifetime but not in schrodinger picture", "read_param");

	if (alg.linearize && alg.ddmdteq)
		error_message("if master equation is linearized, ddmdt_eq is not needed", "read_param");
	if (alg.linearize && !alg.summode)
		error_message("linearization is only supported when we use the generalized scattering-rate matrices P?", "read_param");
	if (alg.linearize && (pmp.laserAlg == "lindblad" || pmp.laserAlg == "coherent"))
		error_message("linearization is not allowed for real-time laser", "read_param");
	if (alg.linearize && (alg.Pin_is_sparse || alg.sparseP))
		error_message("linearization does not support sparse matrices", "read_param");
	if (alg.linearize && alg.linearize_dPee)
		error_message("alg_linearize_dPee and alg.linearize cannot be both true", "read_param");
	if (alg.linearize_dPee && (alg.Pin_is_sparse || alg.sparseP))
		error_message("linearization does not support sparse matrices", "read_param");
	if (alg.linearize_dPee && !alg.ddmdteq)
		error_message("linearize_dPee must be used ddmdteq", "read_param");
	if (eep.eeMode == "none" && alg.linearize_dPee)
		error_message("alg_linearize_dPee is only active when e-e scattering is active", "read_param");
	if (!alg.summode && alg.expt)
		error_message("if alg_summode is false, alg_expt must be false, since this case is not implemented", "read_param");
	if (alg.scatt != "lindblad" && alg.scatt != "conventional")
		error_message("alg_scatt value is not allowed");
	if (alg.scatt == "conventional" && !alg.summode)
		error_message("alg.scatt == \"conventional\" && !alg.summode is not implemented", "read_param");
	if (alg.picture != "interaction" && alg.picture != "schrodinger")
		error_message("alg_picture must be interaction schrodinger or non-interaction", "read_param");
	if (alg.picture == "schrodinger" && (alg.expt || alg.expt_elight))
		error_message("in schrodinger picture, alg_expt and alg_expt_elight must be false", "read_param");
	if (alg.eph_sepr_eh && !alg.eph_need_elec && !alg.eph_need_hole)
		error_message("if alg_eph_sepr_eh, either alg_eph_need_elec or alg_eph_need_hole", "read_param");
	if (alg.ode_method != "rkf45" && alg.ode_method != "euler")
		error_message("alg_ode_method must be rkf45 or euler", "read_param");
	clp.check_params();
	if (alg.only_eimp && eip.ni.size() == 0)
		error_message("alg_only_eimp is only possible if impurity_density is non-zero", "read_param");
	if ((eip.ni.size() != 0 && !alg.scatt_enable) || (eep.eeMode != "none" && !alg.scatt_enable) || (alg.eph_enable && !alg.scatt_enable))
		error_message("alg_scatt_enable must be true if any scattering mechanism is considered", "read_param");
	if (!alg.eph_enable && eip.ni.size() == 0 && eep.eeMode == "none" && alg.scatt_enable)
		error_message("alg_scatt_enable must be false if no scattering mechanism is considered", "read_param");
	if (eip.ni.size() != 0 && !alg.summode)
		error_message("for e-imp, alg.summode is necessary");
	for (int iD = 0; iD < eip.ni.size(); iD++){
		if (clp.scrMode == "none" && eip.impMode[iD] == "model_ionized" && eip.ni[iD] != 0)
			error_message("scrMode should not be none if considering model electron-ionized-impurity scattering", "read_param");
		if (eip.impMode[iD] != "model_ionized" && eip.impMode[iD] != "ab_neutral")
			error_message("impMode must be model_ionized or ab_neutral now", "read_param");
		if (eip.impMode[iD] == "ab_neutral" && alg.sparseP)
			error_message("ab initio neutral e-imp does not support sparse matrices", "read_param");
		if (carrier_density * eip.ni[iD] < 0)
			error_message("currently carrier_denstiy and impurity_density must have the same sign");
		if (eip.Z[iD] <= 0)
			error_message("for ionized impurity, Z is defined as a positive value in this code");
	}
	if (clp.scrMode == "none" && eep.eeMode != "none")
		error_message("scrMode should not be none if considering electron-electron scattering", "read_param");
	if (eep.eeMode != "none" && eep.eeMode != "Pee_fixed_at_eq" && eep.eeMode != "Pee_update")
		error_message("eeMode must be none or Pee_fixed_at_eq now", "read_param");
	if (!alg.linearize && eep.eeMode == "Pee_update" && freq_update_ee_model == 0)
		error_message("want to update Pee but set freq_update_ee_model as 0","read_param");
	if (alg.eph_enable && (alg.only_eimp || alg.only_ee))
		error_message("if only e-imp or only e-e, e-ph should not be enabled. Set alg_eph_enable = 0", "read_param");
	if (alg.only_ee && eep.eeMode == "none")
		error_message("alg_only_ee is only possible if eeMode is not none", "read_param");
	if (alg.only_eimp && eep.eeMode != "none")
		error_message("if you want only e-i scattering, please set eeMode to be none","read_param");
	if (alg.only_intravalley && alg.only_intervalley)
		error_message("only_intravalley and only_intervalley cannot be true at the same time", "read_param");
	if (alg.Pin_is_sparse && !alg.eph_enable){
		bool no_ab_neutral = true;
		for (int iD = 0; iD < eip.ni.size(); iD++)
		if (eip.impMode[iD] == "ab_neutral") no_ab_neutral = false;
		if (no_ab_neutral) error_message("alg_Pin_is_sparse implies ab initio e-ph or e-i is enabled", "read_param");
	}
	if (!alg.linearize && freq_update_eimp_model != freq_update_ee_model)
		error_message("freq_update_eimp_model is the same as freq_update_ee_model in current version", "read_param");

	if (ode.hstart < 0 || ode.hmin < 0 || ode.hmax < 0 || ode.hmax_laser < 0 || ode.epsabs < 0)
		error_message("ode_hstart < 0 || ode_hmin < 0 || ode_hmax < 0 || ode_hmax_laser < 0 || ode_epsabs < 0 is not allowed", "read_param");
	if (ode.hmin > std::max(ode.hmax, ode.hmax_laser) || ode.hstart > std::max(ode.hmax, ode.hmax_laser))
		error_message("ode.hmin > std::max(ode.hmax, ode.hmax_laser) || ode.hstart > std::max(ode.hmax, ode.hmax_laser) is unreasonable", "read_param");

	if (pmp.laserMode != "pump" and pmp.laserMode != "constant")
		error_message("laserMode must be pump or constant in this version", "read_param");
	if (pmp.laserMode != "pump" and pmp.laserAlg == "perturb")
		error_message("laserAlg perturb requires that laserMode must be pump", "read_param");
	if (pmp.laserAlg != "perturb" && pmp.laserAlg != "lindblad" && pmp.laserAlg != "coherent")
		error_message("laserAlg must be perturb or lindblad or coherent in this version", "read_param");
	if (pmp.laserA < 0)
		error_message("laserA must not < 0", "read_param");
	if (pmp.laserA > 0 and pmp.laserE <= 0)
		error_message("if laserA > 0, laserE must be > 0", "read_param");
	if (pmp.laserA > 0 and pmp.pumpTau <= 0)
		error_message("if laserA > 0, pumpTau must be > 0", "read_param");
	if (pmp.laserA > 0 && pmp.laserPoltype == "NONE")
		error_message("laserPoltype == NONE when laserA > 0", "read_param");
	if (pmp.laserA > 0 && Bpert.length() > 1e-12)
		error_message("if laserA > 0, Bpert must be 0", "read_param");

	if (!gfac_k_resolved)
		error_message("currently, g factor should not have band dependence", "read_param");

	if (type_q_ana != "wrap_around_valley" && type_q_ana != "wrap_around_minusk")
		error_message("type_q_ana should be wrap_around_valley or wrap_around_minusk", "read_param");
}
double parameters::get(std::map<std::string, std::string> map, string key, double defaultVal, double unit) const
{
	auto iter = map.find(key);
	if (iter == map.end()) //not found
	{
		if (std::isnan(defaultVal)) //no default provided
		{
			error_message("could not find input parameter");
		}
		else {
			double d = defaultVal * unit;
			if (ionode) printf("%s = %lg\n", key.c_str(), d);
			return d;
		}
	}
	double d = atof(iter->second.c_str()) * unit;
	if (ionode) printf("%s = %lg\n", key.c_str(), d);
	return d;
}
vector3<> parameters::getVector(std::map<std::string, std::string> map, string key, vector3<> defaultVal, double unit) const
{
	auto iter = map.find(key);
	if (iter == map.end()) //not found
	{
		if (std::isnan(defaultVal[0])) //no default provided
		{
			error_message("could not find input parameter");
		}
		else{
			if (ionode) printf("%s = %lg %lg %lg\n", key.c_str(), defaultVal[0] * unit, defaultVal[1] * unit, defaultVal[2] * unit);
			return defaultVal * unit;
		}
	}
	//Parse value string with comma as a delimiter:
	vector3<> result;
	istringstream iss(iter->second);
	for (int k = 0; k<3; k++)
	{
		string token;
		getline(iss, token, ',');
		result[k] = atof(token.c_str()) * unit;
	}
	if (ionode) printf("%s = %lg %lg %lg\n", key.c_str(), result[0], result[1], result[2]);
	return result;
}
string parameters::getString(std::map<std::string, std::string> map, string key, string defaultVal) const
{
	auto iter = map.find(key);
	if (iter == map.end()){ //not found
		if (ionode) printf("%s = %s\n", key.c_str(), defaultVal.c_str());
		return defaultVal;
	}
	if (ionode) printf("%s = %s\n", key.c_str(), (iter->second).c_str());
	return iter->second;
}

std::string parameters::trim(std::string s){
	s.erase(0, s.find_first_not_of(" "));
	return s.erase(s.find_last_not_of(" \n\r\t") + 1);
}

std::map<std::string, std::string> parameters::map_input(fstream& fin){
	std::map<std::string, std::string> param;
	bool b_join = false;
	bool b_joined = false;
	bool b_will_join = false;
	std::string last_key = "";
	for (std::string line; std::getline(fin, line);){
		b_will_join = false;
		b_joined = false;
		string line2 = trim(line);
		//Skip comments
		if (line[0] == '#') continue;
		//Remove "\" 
		if (line2[line2.length() - 1] == '\\'){
			b_will_join = true;
			line2[line2.length() - 1] = ' ';
		}

		if (b_join){
			param[last_key] += " " + line2;
			b_joined = true;
		}
		b_join = b_will_join;

		if (!b_joined){
			size_t equalpos = line2.find('=');
			if (equalpos == std::string::npos) continue;
			last_key = trim(line2.substr(0, equalpos - 1));
			param[last_key] = trim(line2.substr(equalpos + 1, line2.length() - equalpos - 1));
		}
	}
	return param;
}