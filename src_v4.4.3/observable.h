#pragma once
#include "common_headers.h"
#include "parameters.h"

template<class Tl, class Te>
class observable{
public:
	Tl* latt;
	Te* elec;
	int dim;
	double VAL, t0;
	int freq_measure, freq_measure_ene, freq_compute_tau;
	observable(Tl* latt, parameters* param, Te* elec)
		:latt(latt), elec(elec), dim(latt->dim), t0(param->restart ? param->t0 : param->t0 - 1),
		freq_measure(param->freq_measure), freq_measure_ene(param->freq_measure_ene), freq_compute_tau(param->freq_compute_tau)
	{
		if (dim == 3)
			VAL = latt->volume;
		else if (dim == 2)
			VAL = latt->area;
		else if (dim == 1)
			VAL = latt->length;
		else
			VAL = 1.;
	}
};

// assume semiconductor, probably for metal, you just need to ensure nv=0
template<class Tl, class Te>
class ob_1dmk:public observable<Tl, Te>{
public:
	int nk_glob, nb, nv; // nb = nb_dm in class electron
	double prefac;
	double **e, **f;
	complex ***s;
	GaussianSmapling *gauss_elec, *gauss_hole, *gauss_dot_elec, *gauss_dot_hole;
	double **obdot, *obk;
	std::vector<int> ik_kpath;
	complex **ddm_eq, **ddm_neq;
	double trace_sq_ddm_tot, trace_sq_ddmneq_tot, *trace_sq_ddm_eq, *trace_sq_ddm_neq;

	ob_1dmk(Tl* latt, parameters *param, Te* elec)
		:observable<Tl, Te>(latt, param, elec), 
		nk_glob(elec->nk), nb(elec->nb_dm), nv(elec->nv_dm), prefac(1. / elec->nk_full), e(elec->e_dm), f(elec->f_dm),
		ik_kpath(elec->ik_kpath), trace_sq_ddm_tot(elec->trace_sq_ddm_tot), trace_sq_ddmneq_tot(0),
		ddm_eq(elec->ddm_Bpert), ddm_neq(elec->ddm_Bpert_neq)
	{
		if (nb > nv){
			gauss_elec = new GaussianSmapling(elec->ecmin, param->de_measure, elec->emax, param->degauss_measure);
			gauss_dot_elec = new GaussianSmapling(elec->ecmin, param->de_measure, elec->emax, param->degauss_measure);
		}
		if (nv > 0){
			gauss_hole = new GaussianSmapling(elec->emin, param->de_measure, elec->evmax, param->degauss_measure);
			gauss_dot_hole = new GaussianSmapling(elec->emin, param->de_measure, elec->evmax, param->degauss_measure);
		}
		this->s = elec->s;
		obk = new double[nk_glob]; zeros(obk, nk_glob);
		obdot = alloc_real_array(nk_glob, nb);

		if (ddm_eq != nullptr){
			trace_sq_ddm_eq = new double[nk_glob];
			for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
				trace_sq_ddm_eq[ik_glob] = trace_square_hermite(ddm_eq[ik_glob], nb);
			for (int ik = 0; ik < ik_kpath.size(); ik++){
				int ik_glob = ik_kpath[ik];
				if (ionode) printf_complex_mat(ddm_eq[ik_glob], nb, "ddm_eq:");
				if (ionode) printf("trace_sq_ddm_eq = %14.7le\n", trace_sq_ddm_eq[ik_glob]); fflush(stdout);
			}
		}
		if (ddm_neq != nullptr){
			trace_sq_ddm_neq = new double[nk_glob];
			trace_sq_ddmneq_tot = 0;
			for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
				trace_sq_ddm_neq[ik_glob] = trace_square_hermite(ddm_neq[ik_glob], nb);
				trace_sq_ddmneq_tot += trace_sq_ddm_neq[ik_glob];
			}
			for (int ik = 0; ik < ik_kpath.size(); ik++){
				int ik_glob = ik_kpath[ik];
				if (ionode) printf_complex_mat(ddm_neq[ik_glob], nb, "ddm_neq:");
				if (ionode) printf("trace_sq_ddm_neq = %14.7le\n", trace_sq_ddm_neq[ik_glob]); fflush(stdout);
			}
		}
	}

	void measure(string what, string label, bool diff, bool print_ene, double t, complex **dm, complex **ddmdt = nullptr, double dt = 0);
	void measure_brange(string what, bool diff, bool print_ene, double t, complex **dm, complex **ddmdt, double dt, int bStart, int bEnd, string scarr, GaussianSmapling *gauss, GaussianSmapling *gauss_dot);
	void ddmk_ana(double t, complex **dm);
};

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::measure(string what, string label, bool diff, bool print_ene, double t, complex **dm, complex **ddmdt, double dt){
	if (ddmdt && !diff) error_message("ddmdt && !diff");
	if (ddmdt && what == "dos") error_message("ddmdt && dos");
	if (nb > nv)
		measure_brange(what, diff, print_ene, t, dm, ddmdt, dt, nv, nb, "_elec" + label, gauss_elec, gauss_dot_elec);
	if (nv > 0)
		measure_brange(what, diff, print_ene, t, dm, ddmdt, dt, 0, nv, "_hole" + label, gauss_hole, gauss_dot_hole);
}

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::measure_brange(string what, bool diff, bool print_ene, double t, complex **dm, complex **ddmdt, double dt, int bStart, int bEnd, string scarr, GaussianSmapling *gauss, GaussianSmapling *gauss_dot){
	MPI_Barrier(MPI_COMM_WORLD);
	if (!ionode) return;

	bool b_tot = true;
	string fname;
	FILE *fil, *filtot, *filobdotknout, *filobdotknbin;
	if (!diff) scarr = "_initial" + scarr;
	int idir;
	if (what == "sx" || what == "sy" || what == "sz"){
		if (print_ene && !diff) error_message("print_ene && !diff is not allowed currently");
		if (what == "sx") idir = 0;
		else if (what == "sy") idir = 1;
		else if (what == "sz") idir = 2;
		if (!ddmdt && diff){
			fname = what + scarr + "_ene.out"; fil = fopen(fname.c_str(), "a");
			fname = what + scarr + "_tot.out";  filtot = fopen(fname.c_str(), "a");
		}
		else if(ddmdt){
			fname = what + "Dot" + scarr + "_ene.out"; fil = fopen(fname.c_str(), "a");
			fname = "tau_" + what + scarr + "_tot.out";  filtot = fopen(fname.c_str(), "a");
			//fname = "obdotkn_" + what + scarr + ".out";  filobdotknout = fopen(fname.c_str(), "w");
			//fname = "obdotkn_" + what + scarr + ".bin";  filobdotknbin = fopen(fname.c_str(), "w");
		}
	}
	if (what == "dos" || what == "fn") b_tot = false;
	if (what == "dos") { fname = what + scarr + ".out"; fil = fopen(fname.c_str(), "a"); }
	if (what == "fn"){
		if (!ddmdt){
			fname = what + scarr + ".out"; fil = fopen(fname.c_str(), "a");
		}
		else{
			fname = what + "Dot" + scarr + ".out"; fil = fopen(fname.c_str(), "a");
			//fname = "obdotkn_" + what + scarr + ".out";  filobdotknout = fopen(fname.c_str(), "w");
			//fname = "obdotkn_" + what + scarr + ".bin";  filobdotknbin = fopen(fname.c_str(), "w");
		}
	}

	double tot = 0, tot_amp = 0, dottot = 0, dottot_amp = 0, tot_tplusdt = 0, dottot_term2 = 0;
	zeros(obk, nk_glob);
	if (ddmdt) zeros(obdot, nk_glob, nb);
	gauss->reset();
	if (ddmdt) gauss_dot->reset();

	if (print_ene) fprintf(fil, "**************************************************\n");
	if (!ddmdt && print_ene) fprintf(fil, "time = %10.3e, print energy and ob(e)\n", t);
	if (ddmdt && print_ene) fprintf(fil, "time = %10.3e, print energy and obdot(e)\n", t);
	if (print_ene) fprintf(fil, "--------------------------------------------------\n");

	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		//bool isK = this->latt->isKvalley(this->elec->kvec[ik_glob]);
		//if ((what == "sxk" || what == "syk") && (!isK)) continue;

		for (int i = bStart; i < bEnd; i++){
			double ob = 0, ob_amp = 0, dot = 0, dot_amp = 0, ob_tplusdt = 0, dot_term2 = 0;
			if (what == "dos")
				ob = 1;
			else if (what == "fn"){
				if (diff)
					ob = real(dm[ik_glob][i*nb + i] - f[ik_glob][i]);
				else
					ob = f[ik_glob][i];
				if (ddmdt)
					dot = real(ddmdt[ik_glob][i*nb + i]);
			}
			else{
				//for (int b = bStart; b < bEnd; b++){
				for (int b = 0; b < nb; b++){
					complex ob_kib = s[ik_glob][idir][i*nb + b];
					complex ob_kib_t = (i == b || alg.picture == "schrodinger") ? ob_kib : (ob_kib * cis((e[ik_glob][i] - e[ik_glob][b])*t));
					complex dm_kbi = (i == b) ? (dm[ik_glob][b*nb + i] - f[ik_glob][i]) : dm[ik_glob][b*nb + i];
					if (diff){
						ob += real(ob_kib_t * dm_kbi);
						ob_amp += real(ob_kib * dm_kbi);
					}
					else{
						if (i == b) ob += real(ob_kib * f[ik_glob][i]);
					}
					if (ddmdt){
						dot += real(ob_kib_t * ddmdt[ik_glob][b*nb + i]);
						dot_amp += real(ob_kib * ddmdt[ik_glob][b*nb + i]);
						complex ctmp = ob_kib_t * dm_kbi;
						if (alg.picture == "interaction") dot_term2 -= 2 * imag(e[ik_glob][i] * ctmp);
						if (dt > 0){
							complex ob_kib_tplusdt = (i == b || alg.picture == "schrodinger") ? ob_kib : (ob_kib * cis((e[ik_glob][i] - e[ik_glob][b])*(t + dt)));
							complex dm_tplusdt = dm[ik_glob][b*nb + i] + ddmdt[ik_glob][b*nb + i] * dt;
							if (i == b) dm_tplusdt -= f[ik_glob][i];
							ob_tplusdt += real(ob_kib_tplusdt * dm_tplusdt);
						}
					}
				}
			}
			obk[ik_glob] += ob;
			tot += ob; tot_amp += ob_amp; tot_tplusdt += ob_tplusdt;
			dottot += dot; dottot_amp += dot_amp; dottot_term2 += dot_term2;
			obdot[ik_glob][i] = dot;

			double ene = e[ik_glob][i];
			gauss->addEvent(ene, ob);
			gauss->addEvent2(ene, ob_amp);
			if (ddmdt) gauss_dot->addEvent(ene, dot);
			if (ddmdt) gauss_dot->addEvent2(ene, dot_amp);
			if (ddmdt && dt > 0) gauss_dot->addEvent3(ene, (ob_tplusdt - ob) / dt);
		}
	}

	tot *= prefac; tot_amp *= prefac; tot_tplusdt *= prefac;
	dottot *= prefac; dottot_amp *= prefac; dottot_term2 *= prefac;
	double dens, au2cm = 5.291772109038e-9;
	if (this->dim == 3)
		dens = tot / (this->VAL*std::pow(au2cm, 3));
	else if (this->dim == 2)
		dens = tot / (this->VAL*std::pow(au2cm, 2));
	else if (this->dim == 1)
		dens = tot / (this->VAL*au2cm);
	else
		dens = tot;
	if (b_tot){
		if (!ddmdt){
			if (diff){
				if (fabs(t) < 1e-30)
					fprintf(filtot, " 0.00000000000000e+01 %21.14le %21.14le\n", tot, tot_amp);
				else
					fprintf(filtot, "%21.14le %21.14le %21.14le\n", t, tot, tot_amp);
			}
			else
				printf("\ninitial %s %21.14le\n", what.c_str(), tot);
		}
		else{
			double tautot_fd = (dt == 0) ? 0 : -tot * dt / (tot_tplusdt - tot);
			if (alg.picture == "interaction"){
				if (fabs(t) < 1e-30) fprintf(filtot, " 0.0000000000e+01 fs: %14.7le %14.7le %14.7le ps\n", tautot_fd / ps, -tot / dottot / ps, -tot / (dottot + dottot_term2) / ps); // print tau_tot
				else fprintf(filtot, "%17.10le fs: %14.7le %14.7le %14.7le ps\n", t / fs, tautot_fd / ps, -tot / dottot / ps, -tot / (dottot + dottot_term2) / ps); // print tau_tot
			}
			else{
				if (fabs(t) < 1e-30) fprintf(filtot, " 0.0000000000e+01 fs: %14.7le ps\n", -tot / dottot / ps); // print tau_tot
				else fprintf(filtot, "%17.10le fs: %14.7le ps\n", t / fs, -tot / dottot / ps); // print tau_tot
			}
		}
	}
	if (b_tot && diff) fflush(filtot);
	if (b_tot && diff) fclose(filtot);
	if (!ddmdt && print_ene && !b_tot) gauss->print(fil, 1.0, prefac);
	if (!ddmdt && print_ene && b_tot) gauss->print2(fil, 1.0, prefac);
	if (ddmdt && print_ene && !b_tot) gauss_dot->print(fil, 1.0, prefac);
	if (ddmdt && print_ene && b_tot) gauss_dot->print2(fil, 1.0, prefac);
	if (ddmdt && print_ene && b_tot) gauss_dot->print3(fil, 1.0, prefac);
	if (!ddmdt && print_ene && !b_tot){
		if (fabs(t) < 1e-30)
			fprintf(fil, "time and density:  0.00000000000000e+01 %21.14le\n", dens);
		else
			fprintf(fil, "time and density:  %21.14le %21.14le\n", t, dens); // carrier density in fn_ene.out
	}
	if (print_ene) fprintf(fil, "**************************************************\n");
	//if (ddmdt){
	//	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
	//	for (int i = bStart; i < bEnd; i++)
	//		fprintf(filobdotknout, "%14.7le %14.7le\n", e[ik_glob][i], obdot[ik_glob][i]);
	//	fwrite(obdot, sizeof(double), nk_glob * nb, filobdotknbin);
	//	fclose(filobdotknout); fclose(filobdotknbin);
	//}
	if (!b_tot || diff) fflush(fil); fflush(stdout);
	if (!b_tot || diff) fclose(fil);
}

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::ddmk_ana(double t, complex **dm){
	if (ddm_eq == nullptr || ddm_neq == nullptr) return;
	MPI_Barrier(MPI_COMM_WORLD);
	if (!ionode) return;
	string dir = "ddm_along_kpath_results/";
	complex *ddm = new complex[nb*nb];
	complex *ddm_res = new complex[nb*nb];
	complex *ddm_norm = new complex[nb*nb];
	string fname = dir + "ratio_tauneq_taufm.dat"; FILE *filratio = fopen(fname.c_str(), "wb");

	double trace_AB_0_tot = 0, trace_AB_1_tot = 0, trace_sq_res_tot = 0;
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		int ik_along_kpath = -1;
		for (int ik = 0; ik < ik_kpath.size(); ik++)
			if (ik_kpath[ik] == ik_glob) { ik_along_kpath = ik; break; }

		ostringstream oss; oss << ik_along_kpath;
		string str_ik = oss.str();

		// ddm in schrodinger picture
		for (int i = 0; i < nb; i++)
		for (int b = 0; b < nb; b++)
		if (i == b)
			ddm[i*nb + b] = dm[ik_glob][i*nb + b] - f[ik_glob][i];
		else
			ddm[i*nb + b] = (alg.picture == "schrodinger") ? dm[ik_glob][i*nb + b] : dm[ik_glob][i*nb + b] * cis((e[ik_glob][b] - e[ik_glob][i])*t);
		
		// normalization
		double trace_sq_ddm = trace_square_hermite(ddm, nb);
		for (int ibb = 0; ibb < nb*nb; ibb++)
			ddm_norm[ibb] = ddm[ibb] / sqrt(trace_sq_ddm);

		if (ik_along_kpath >= 0){
			string fname = dir + "ddm_norm_k" + str_ik + ".out"; FILE *filddm = fopen(fname.c_str(), "a");
			fprintf(filddm, "%14.7le fs:", t / fs);
			fprintf_complex_mat(filddm, ddm_norm, nb, "");
			fclose(filddm);
		}

		// compute ddm_eq composition of ddm, Tr{ddm ddm_eq} / Tr{ddm_eq ddm_eq}
		double trace_AB_0 = trace_AB(ddm, ddm_eq[ik_glob], nb);
		trace_AB_0_tot += trace_AB_0;
		double weight_ddm_eq = trace_AB_0 / trace_sq_ddm_eq[ik_glob];
		// compute ddm_neq composition of ddm, Tr{ddm ddm_neq} / Tr{ddm_neq ddm_neq}
		double trace_AB_1 = trace_AB(ddm, ddm_neq[ik_glob], nb);
		trace_AB_1_tot += trace_AB_1;
		double weight_ddm_neq = trace_AB_1 / trace_sq_ddm_neq[ik_glob];
		for (int ibb = 0; ibb < nb*nb; ibb++)
			ddm_res[ibb] = ddm[ibb] - weight_ddm_eq * ddm_eq[ik_glob][ibb] - weight_ddm_neq * ddm_neq[ik_glob][ibb];
		double trace_sq_res = trace_square_hermite(ddm_res, nb);
		trace_sq_res_tot += trace_sq_res;

		if (ik_along_kpath >= 0){
			double trace_AB_res_0 = trace_AB(ddm_res, ddm_eq[ik_glob], nb);
			double trace_AB_res_1 = trace_AB(ddm_res, ddm_neq[ik_glob], nb);
			string fname = dir + "ddm_decomp_k" + str_ik + ".out"; FILE *fildecomp = fopen(fname.c_str(), "a");
			fprintf(fildecomp, "%14.7le fs: %14.7le %14.7le %14.7le %14.7le %14.7le %14.7le\n", 
				t / fs, sqrt(trace_sq_ddm), weight_ddm_eq, weight_ddm_neq, sqrt(trace_sq_res), trace_AB_res_0, trace_AB_res_1);
			fclose(fildecomp);

			for (int ibb = 0; ibb < nb*nb; ibb++)
				ddm_norm[ibb] = ddm_res[ibb] / sqrt(trace_sq_res);

			fname = dir + "ddm_res_norm_k" + str_ik + ".out"; FILE *filddmres = fopen(fname.c_str(), "a");
			fprintf(filddmres, "%14.7le fs:", t / fs);
			fprintf_complex_mat(filddmres, ddm_norm, nb, "");
			fclose(filddmres);
		}

		double ratio = weight_ddm_neq / weight_ddm_eq;
		fwrite(&ratio, sizeof(double), 1, filratio);
	}
	fclose(filratio);

	fname = dir + "ddm_decomp_avg.out"; FILE *fildecompavg = fopen(fname.c_str(), "a");
	fprintf(fildecompavg, "%14.7le fs: %14.7le %14.7le %14.7le %14.7le\n", t / fs, sqrt(trace_sq_ddm_tot/nk_glob), 
		trace_AB_0_tot / trace_sq_ddm_tot, trace_AB_1_tot / trace_sq_ddmneq_tot, sqrt(trace_sq_res_tot/nk_glob));
	fclose(fildecompavg);
}