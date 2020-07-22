#include "parameters.h"
#include "mymp.h"
#include "ElecLight.h"
#include "ODE.h"

void parameters::read_jdftx(){
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	for (int i = 0; i < 5; i++)
		fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		sscanf(s, "%le", &temperature); if (ionode) printf("temperature = %14.7le\n", temperature);
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

	restart = get(param_map, "restart", false);
	if (ionode && !restart && is_dir("restart"))
		error_message("diretory restart presents, you should run a restart calculation");
	if (ionode && !restart) system("mkdir restart");
	MPI_Barrier(MPI_COMM_WORLD);
	compute_dot_only = get(param_map, "compute_dot_only", false);

	// algorithm parameters
	code = getString(param_map, "code", "mos2");
	alg.eph_enable = get(param_map, "alg_eph_enable", true);
	alg.H0hasBS = get(param_map, "alg_H0hasBS", true);
	alg.summode = get(param_map, "alg_summode", 1);
	alg.ddmeq = get(param_map, "alg_ddmeq", 0);
	alg.expt = get(param_map, "alg_expt", 1);
	alg.expt_elight = get(param_map, "alg_expt_elight", alg.expt);
	alg.scatt = getString(param_map, "alg_scatt", "lindblad");
	alg.picture = getString(param_map, "alg_picture", "interaction");

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

	alg.sparseP = get(param_map, "alg_sparseP", 0);
	alg.thr_sparseP = get(param_map, "alg_thr_sparseP", 1e-40);

	alg.ode_method = getString(param_map, "alg_ode_method", "rkf45");

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
	de_measure = get(param_map, "de_measure", 5e-4, eV);
	degauss_measure = get(param_map, "degauss_measure", 2e-3, eV);
	mu = get(param_map, "mu", 0., eV);
	if (code == "jdftx")
		read_jdftx();
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
	if (code == "jdftx"){
		if (!alg.summode)
			error_message("if code is jdftx, alg_summode must be true");
		if (!alg.H0hasBS && B.length() < 1e-10)
			error_message("it is declared there is BS coherent term but |B| is zero");
		if (alg.H0hasBS && B.length() > 1e-10)
			error_message("if H0 has already BS, please do not set Bx, By and Bz");
	}
	if (code == "mos2"){
		if (!alg.eph_sepr_eh)
			error_message("if code is mos2, alg_eph_sepr_eh must be true");
		if (pmp.pumpA0 > 0)
			error_message("for mos2 model, pump is not allowed");
	}

	if (!alg.summode && alg.expt)
		error_message("if alg_summode is false, alg_expt must be false, since this case is not implemented");
	if (alg.scatt != "lindblad" && alg.scatt != "conventional")
		error_message("alg_scatt value is not allowed");
	if (alg.scatt == "conventional" && !alg.summode)
		error_message("alg.scatt == \"conventional\" && !alg.summode is not implemented");
	if (alg.picture != "interaction")
		error_message("alg_picture must be interaction now");
	if (alg.eph_sepr_eh && !alg.eph_need_elec && !alg.eph_need_hole)
		error_message("if alg_eph_sepr_eh, either alg_eph_need_elec or alg_eph_need_hole");
	if (alg.ode_method != "rkf45" && alg.ode_method != "euler")
		error_message("alg_ode_method must be rkf45 or euler");

	if (ode.hstart < 0 || ode.hmin < 0 || ode.hmax < 0 || ode.hmax_pump < 0 || ode.epsabs < 0)
		error_message("ode_hstart < 0 || ode_hmin < 0 || ode_hmax < 0 || ode_hmax_pump < 0 || ode_epsabs < 0 is not allowed");
	if (ode.hmin > std::max(ode.hmax, ode.hmax_pump) || ode.hstart > std::max(ode.hmax, ode.hmax_pump))
		error_message("ode.hmin > std::max(ode.hmax, ode.hmax_pump) || ode.hstart > std::max(ode.hmax, ode.hmax_pump) is unreasonable");

	if (pumpMode != "perturb" && pumpMode != "lindblad" && pumpMode != "coherent")
		error_message("pumpMode must be perturb or lindblad or coherent in this version");
	if (pmp.pumpA0 < 0)
		error_message("pumpA0 must not < 0");
	if (pmp.pumpA0 > 0 && pmp.pumpPoltype == "NONE")
		error_message("pumpPoltype == NONE when pumpA0 > 0");
	if (pmp.pumpA0 > 0 && Bpert.length() > 1e-10)
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
			if (ionode) printf("%s = %lg %lg %lg\n", key.c_str(), defaultVal[0], defaultVal[1], defaultVal[2]);
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
		result[k] = atof(token.c_str());
	}
	if (ionode) printf("%s = %lg %lg %lg\n", key.c_str(), result[0], result[1], result[2]);
	return result * unit;
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