#include "parameters.h"
#include "mymp.h"
#include "PumpProbe.h"
#include "ScattModel_Param.h"
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

void parameters::read_param(){
	fstream fin;
	fin.open("param.in", ios::in);
	if (fin.fail()) error_message("input file param.in does not exist");
	std::map<std::string, std::string> param_map = map_input(fin);

	if (ionode) printf("\n");
	if (ionode) printf("reading parameters\n");
	
	DEBUG = get(param_map, "DEBUG", false);
	if (ionode && DEBUG && !is_dir("debug_info")) system("mkdir debug_info");
	restart = get(param_map, "restart", false);
	if (ionode && !restart && is_dir("restart"))
		error_message("diretory restart presents, you should run a restart calculation");
	if (ionode && !restart){
		if (is_dir("ddm_along_kpath_results")) system("rm -r ddm_along_kpath_results");
		system("mkdir ddm_along_kpath_results");
	}
	if (ionode && !restart) system("mkdir restart");
	MPI_Barrier(MPI_COMM_WORLD);
	compute_tau_only = get(param_map, "compute_tau_only", false);

	alg.use_dmDP_taufm_as_init = get(param_map, "alg_use_dmDP_taufm_as_init", false);
	alg.DP_beyond_carrierlifetime = get(param_map, "alg_DP_beyond_carrierlifetime", false);
	alg.mix_tauneq = get(param_map, "alg_mix_tauneq", 0.2);
	alg.positive_tauneq = get(param_map, "alg_positive_tauneq", false);
	alg.use_dmDP_in_evolution = get(param_map, "alg_use_dmDP_in_evolution", false);

	print_along_kpath = get(param_map, "print_along_kpath", false);
	while (print_along_kpath){
		int iPath = int(kpath_start.size()) + 1;
		ostringstream oss; oss << iPath;
		string pathName = oss.str();
		vector3<double> kstart = getVector(param_map, "kpath_start" + pathName, vector3<>(2,2,2));
		vector3<double> kend = getVector(param_map, "kpath_end" + pathName, vector3<>(2,2,2));
		if (kstart == vector3<>(2,2,2) || kend == vector3<>(2,2,2)) break;
		if (kend == kstart) error_message("kstart == kend");
		kpath_start.push_back(kstart);
		kpath_end.push_back(kend);
	}
	if (print_along_kpath && kpath_start.size() == 0) error_message("print_along_kpath && kpath_start.size() == 0");

	// algorithm parameters
	code = getString(param_map, "code", "mos2");
	alg.eph_enable = get(param_map, "alg_eph_enable", true);
	alg.only_eimp = get(param_map, "alg_only_eimp", false);
	alg.only_ee = get(param_map, "alg_only_ee", false);
	alg.only_intravalley = get(param_map, "alg_only_intravalley", false);
	alg.only_intervalley = get(param_map, "alg_only_intervalley", false);
	alg.summode = get(param_map, "alg_summode", 1);
	alg.ddmeq = get(param_map, "alg_ddmeq", 0);
	alg.expt = get(param_map, "alg_expt", 1);
	alg.expt_elight = get(param_map, "alg_expt_elight", alg.expt);
	alg.scatt = getString(param_map, "alg_scatt", "lindblad");
	alg.picture = getString(param_map, "alg_picture", "interaction");
	if (alg.picture == "non-interaction") alg.picture = "schrodinger";

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

	alg.phenom_relax = get(param_map, "alg_phenom_relax", 0);
	if (alg.phenom_relax){
		tau_phenom = get(param_map, "tau_phenom", 1e15, ps);
		bStart_tau = get(param_map, "bStart_tau", 0); // relative to bStart_dm
		bEnd_tau = get(param_map, "bEnd_tau", 0);
	}

	// model screened coulomb potential for e-i and e-e scatterings
	clp.scrMode = getString(param_map, "scrMode", "none");
	clp.scrFormula = getString(param_map, "scrFormula", "RPA");
	clp.ovlp = clp.scrFormula == "RPA" ? true : false;
	clp.update = get(param_map, "update_screening", 1);
	clp.dynamic = getString(param_map, "dynamic_screening", "static");
	clp.eppa = get(param_map, "eppa_screening", 0);
	clp.meff = get(param_map, "meff_screening", 1);
	clp.omegamax = get(param_map, "omegamax_screening", 0, eV);
	clp.nomega = get(param_map, "nomega_screening", 1);
	if (clp.dynamic == "ppa") clp.nomega = 2;
	clp.fderavitive_technique = get(param_map, "fderavitive_technique_static_screening", 1);
	clp.smearing = get(param_map, "smearing_screening", 0, eV);
	clp.eps = get(param_map, "epsilon_background", 1);

	// to turn on electron-impurity scattering, set eip.ni nonzero
	eip.ni = get(param_map, "impurity_density", 0, std::pow(bohr2cm, 3));
	eip.impMode = getString(param_map, "impMode", "model_ionized");
	eip.partial_ionized = get(param_map, "partial_ionized", 0);
	eip.Z = get(param_map, "Z_impurity", 1);
	eip.g = get(param_map, "g_impurity", 2);
	eip.lng = std::log(eip.g); if (ionode) printf("eip.lng = %lg\n", eip.lng);
	freq_update_eimp_model = get(param_map, "freq_update_eimp_model", 0);

	eep.eeMode = getString(param_map, "eeMode", "none"); // "none" will turn off electron-electron scattering
	eep.antisymmetry = get(param_map, "ee_antisymmetry", 0);
	freq_update_ee_model = get(param_map, "freq_update_ee_model", 0);

	alg.Pin_is_sparse = get(param_map, "alg_Pin_is_sparse", 0);
	alg.sparseP = get(param_map, "alg_sparseP", 0);
	alg.thr_sparseP = get(param_map, "alg_thr_sparseP", 1e-40);

	alg.ode_method = getString(param_map, "alg_ode_method", "rkf45");

	alg.set_scv_zero = get(param_map, "alg_set_scv_zero", 0);

	// time paramters and studying system parameters
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
	freq_measure_ene = get(param_map, "freq_measure_ene", 10);
	freq_compute_tau = get(param_map, "freq_compute_tau", freq_measure_ene);
	de_measure = get(param_map, "de_measure", 5e-4, eV);
	degauss_measure = get(param_map, "degauss_measure", 2e-3, eV);
	double mu_input = 0;
	double degauss_input = 0, ndegauss_input = 0;
	if (code == "jdftx"){
		read_jdftx();
		mu_input = mu;
		degauss_input = degauss;
		ndegauss_input = ndegauss;
	}
	else{
		temperature = get(param_map, "temperature", 300, Kelvin);
		nk1 = get(param_map, "nk1", 1);
		nk2 = get(param_map, "nk2", 1);
		nk3 = get(param_map, "nk3", 1);
		ewind = get(param_map, "ewind", 6. * temperature);
		degauss = get(param_map, "degauss", 0.01, eV);
		lattvec1 = getVector(param_map, "lattvec1", vector3<>(1., 0., 0.));
		lattvec2 = getVector(param_map, "lattvec2", vector3<>(0., 1., 0.));
		lattvec3 = getVector(param_map, "lattvec3", vector3<>(0., 0., 1.));
		matrix3<> Rtmp(lattvec1[0], lattvec2[0], lattvec3[0],
			lattvec1[1], lattvec2[1], lattvec3[1],
			lattvec1[2], lattvec2[2], lattvec3[2]);
		R = Rtmp;
	}
	mu = get(param_map, "mu", mu_input/eV, eV);
	degauss = get(param_map, "degauss", degauss_input / eV, eV);
	ndegauss = get(param_map, "ndegauss", ndegauss_input);
	eip.Eimp = get(param_map, "E_impurity", mu, eV);
	carrier_density = get(param_map, "carrier_density", 0, std::pow(bohr2cm, 3));
	dim = get(param_map, "dim", 3);
	if (dim == 2)
		thickness = get(param_map, "thickness", 0);

	// magnetic field
	Bx = get(param_map, "Bx", 0., Tesla2au);
	By = get(param_map, "By", 0., Tesla2au);
	Bz = get(param_map, "Bz", 0., Tesla2au);
	B[0] = Bx; B[1] = By; B[2] = Bz;

	// pump and probe
	pumpMode = getString(param_map, "pumpMode", "perturb");
	if (pumpMode == "lindblad" || pumpMode == "coherent")
		tstep_pump = get(param_map, "tstep_pump", tstep / fs, fs);
	else
		tstep_pump = tstep;
	pmp.pumpA0 = get(param_map, "pumpA0", 0.);
	pmp.pumpE = get(param_map, "pumpE", 0., eV);
	pmp.pumpTau = get(param_map, "pumpTau", 0., fs);
	if (!restart){
		pmp.pump_tcenter = get(param_map, "pump_tcenter", (t0 + 5 * pmp.pumpTau) / fs, fs); // 5*Tau is quite enough
		FILE *filtime = fopen("restart/pump_tcenter.dat", "w"); fprintf(filtime, "%14.7le", pmp.pump_tcenter); fclose(filtime);
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
	pmp.pumpPoltype = getString(param_map, "pumpPoltype", "NONE");
	pmp.pumpPol = pmp.set_Pol(pmp.pumpPoltype);
	if (ionode) { pmp.print(pmp.pumpPol); }
	while (true){
		int iPol = int(pmp.probePol.size()) + 1;
		ostringstream oss; oss << iPol;
		string polName = oss.str();
		string poltype = getString(param_map, "probePoltype"+polName, "NONE");
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

	// magnetic field perturbation
	Bxpert = get(param_map, "Bxpert", 0., Tesla2au);
	Bypert = get(param_map, "Bypert", 0., Tesla2au);
	Bzpert = get(param_map, "Bzpert", 0., Tesla2au);
	Bpert[0] = Bxpert; Bpert[1] = Bypert; Bpert[2] = Bzpert;

	// ODE (ordinary derivative equation) parameters
	ode.hstart = get(param_map, "ode_hstart", 1e-3, fs);
	ode.hmin = get(param_map, "ode_hmin", 0, fs);
	ode.hmax = get(param_map, "ode_hmax", std::max(tstep, tstep_pump) / fs, fs);
	double dtmp = pumpMode == "coherent" ? 1 : tstep_pump / fs;
	ode.hmax_pump = get(param_map, "ode_hmax_pump", dtmp, fs);
	ode.epsabs = get(param_map, "ode_epsabs", 1e-8);
	if (ionode) printf("\n");

	// check
	if (code != "mos2" && code != "jdftx")
		error_message("code value is not allowed");
	if (code == "jdftx" && !alg.summode)
		error_message("if code is jdftx, alg_summode must be true");
	if (code == "mos2"){
		if (!alg.eph_sepr_eh)
			error_message("if code is mos2, alg_eph_sepr_eh must be true");
		if (pmp.pumpA0 > 0)
			error_message("for mos2 model, pump is not allowed");
	}
	if (alg.DP_beyond_carrierlifetime && restart)
		error_message("alg_DP_beyond_carrierlifetime && param->restart");
	if (alg.DP_beyond_carrierlifetime && !compute_tau_only)
		error_message("alg_DP_beyond_carrierlifetime && !compute_tau_only");
	if (alg.DP_beyond_carrierlifetime && Bpert.length() < 1e-12)
		error_message("alg_DP_beyond_carrierlifetime && Bpert.length() < 1e-12");
	if (alg.DP_beyond_carrierlifetime && alg.picture != "schrodinger")
		error_message("alg_DP_beyond_carrierlifetime but not in schrodinger picture");

	if (!alg.summode && alg.expt)
		error_message("if alg_summode is false, alg_expt must be false, since this case is not implemented");
	if (alg.scatt != "lindblad" && alg.scatt != "conventional")
		error_message("alg_scatt value is not allowed");
	if (alg.scatt == "conventional" && !alg.summode)
		error_message("alg.scatt == \"conventional\" && !alg.summode is not implemented");
	if (alg.picture != "interaction" && alg.picture != "schrodinger")
		error_message("alg_picture must be interaction schrodinger or non-interaction");
	if (alg.picture == "schrodinger" && (alg.expt || alg.expt_elight))
		error_message("in schrodinger picture, alg_expt and alg_expt_elight must be false");
	if (alg.eph_sepr_eh && !alg.eph_need_elec && !alg.eph_need_hole)
		error_message("if alg_eph_sepr_eh, either alg_eph_need_elec or alg_eph_need_hole");
	if (alg.ode_method != "rkf45" && alg.ode_method != "euler")
		error_message("alg_ode_method must be rkf45 or euler");
	clp.check_params();
	if (clp.scrMode == "none" && eip.impMode == "model_ionized" && eip.ni != 0)
		error_message("scrMode should not be none if considering model electron-ionized-impurity scattering");
	if (clp.scrMode == "none" && eep.eeMode != "none")
		error_message("scrMode should not be none if considering electron-electron scattering", "read_param");
	if (eip.impMode != "model_ionized")
		error_message("impMode must be model_ionized now");
	if (eep.eeMode != "none" && eep.eeMode != "Pee_fixed_at_eq" && eep.eeMode != "Pee_update")
		error_message("eeMode must be none or Pee_fixed_at_eq now");
	if (eep.eeMode == "Pee_update" && freq_update_ee_model == 0)
		error_message("want to update Pee but set freq_update_ee_model as 0","read_param");
	if (alg.only_eimp && eip.ni == 0)
		error_message("alg_only_eimp is only possible if impurity_density is non-zero");
	if (alg.only_ee && eep.eeMode == "none")
		error_message("alg_only_ee is only possible if eeMode is not none");
	if (alg.only_eimp && !alg.eph_enable)
		error_message("current e-i must work with e-ph");
	if (alg.only_eimp && eep.eeMode != "none")
		error_message("if you want only e-i scattering, please set eeMode to be none","read_param");
	if (alg.only_intravalley && alg.only_intervalley)
		error_message("only_intravalley and only_intervalley cannot be true at the same time", "read_param");
	if (freq_update_eimp_model != freq_update_ee_model)
		error_message("freq_update_eimp_model is the same as freq_update_ee_model in current version", "read_param");

	if (ode.hstart < 0 || ode.hmin < 0 || ode.hmax < 0 || ode.hmax_pump < 0 || ode.epsabs < 0)
		error_message("ode_hstart < 0 || ode_hmin < 0 || ode_hmax < 0 || ode_hmax_pump < 0 || ode_epsabs < 0 is not allowed");
	if (ode.hmin > std::max(ode.hmax, ode.hmax_pump) || ode.hstart > std::max(ode.hmax, ode.hmax_pump))
		error_message("ode.hmin > std::max(ode.hmax, ode.hmax_pump) || ode.hstart > std::max(ode.hmax, ode.hmax_pump) is unreasonable");

	if (carrier_density * eip.ni < 0)
		error_message("currently carrier_denstiy and impurity_density must have the same sign");
	if (eip.Z <= 0)
		error_message("for ionized impurity, Z is defined as a positive value in this code");

	if (pumpMode != "perturb" && pumpMode != "lindblad" && pumpMode != "coherent")
		error_message("pumpMode must be perturb or lindblad or coherent in this version");
	if (pmp.pumpA0 < 0)
		error_message("pumpA0 must not < 0");
	if (pmp.pumpA0 > 0 && pmp.pumpPoltype == "NONE")
		error_message("pumpPoltype == NONE when pumpA0 > 0");
	if (pmp.pumpA0 > 0 && Bpert.length() > 1e-12)
		error_message("if pumpA0 > 0, Bpert must be 0");
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