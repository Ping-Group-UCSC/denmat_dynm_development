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
	int freq_measure, freq_measure_ene, freq_compute_tau, it_start_ddm;
	bool print_tot_band;
	observable(Tl* latt, parameters* param, Te* elec)
		:latt(latt), elec(elec), dim(latt->dim), t0(param->restart ? param->t0 : param->t0 - 1),
		freq_measure(param->freq_measure), freq_measure_ene(param->freq_measure_ene), freq_compute_tau(param->freq_compute_tau),
		print_tot_band(param->print_tot_band)
	{
		if (dim == 3)
			VAL = latt->volume;
		else if (dim == 2)
			VAL = latt->area;
		else if (dim == 1)
			VAL = latt->length;
		else
			VAL = 1.;
		it_start_ddm = last_file_index("ddm_along_kpath_results/ddm.", ".dat");
		if (ionode) { printf("\nit_start_ddm = %d\n", it_start_ddm); fflush(stdout); }
	}
};

// assume semiconductor, probably for metal, you just need to ensure nv=0
template<class Tl, class Te>
class ob_1dmk:public observable<Tl, Te>{
public:
	int nk_glob, nb, nv, bStart_eph, bEnd_eph; // nb = nb_dm in class electron
	double prefac;
	double **e, **f;
	complex ***s;
	GaussianSmapling *gauss_elec, *gauss_hole, *gauss_dot_elec, *gauss_dot_hole;
	double *obk, *tot_band, *tot_valley, **tot_valley_band;
	std::vector<int> ik_kpath;
	complex **ddm_eq, **ddm_neq;
	double trace_sq_ddm_tot, trace_sq_ddmneq_tot, *trace_sq_ddm_eq, *trace_sq_ddm_neq;

	ob_1dmk(Tl* latt, parameters *param, Te* elec, int bStart_eph, int bEnd_eph)
		:observable<Tl, Te>(latt, param, elec), 
		nk_glob(elec->nk), nb(elec->nb_dm), nv(elec->nv_dm), prefac(1. / elec->nk_full), e(elec->e_dm), f(elec->f_dm),
		ik_kpath(elec->ik_kpath), trace_sq_ddm_tot(elec->trace_sq_ddm_tot), trace_sq_ddmneq_tot(0),
		ddm_eq(elec->ddm_Bpert), ddm_neq(elec->ddm_Bpert_neq), bStart_eph(bStart_eph), bEnd_eph(bEnd_eph)
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
		tot_band = new double[nb]; zeros(tot_band, nb);
		tot_valley = new double[(int)latt->vpos.size()]; zeros(tot_valley, (int)latt->vpos.size());
		tot_valley_band = alloc_real_array((int)latt->vpos.size(), nb);

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
	void ddmk_ana_drive(int it, double t, complex **dm);
	void ddmk_ana(int it, double t, complex **dm);
};

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::measure(string what, string label, bool diff, bool print_ene, double t, complex **dm, complex **ddmdt, double dt){
	if (ddmdt && !diff) error_message("ddmdt && !diff");
	if (ddmdt && (what == "dos" || what == "fn")) error_message("ddmdt && (dos || fn)");
	MPI_Barrier(MPI_COMM_WORLD);
	if (ionode && nb > nv)
		measure_brange(what, diff, print_ene, t, dm, ddmdt, dt, nv, nb, "_elec" + label, gauss_elec, gauss_dot_elec);
	MPI_Barrier(MPI_COMM_WORLD);
	if (ionode && nv > 0)
		measure_brange(what, diff, print_ene, t, dm, ddmdt, dt, 0, nv, "_hole" + label, gauss_hole, gauss_dot_hole);
	MPI_Barrier(MPI_COMM_WORLD);
}

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::measure_brange(string what, bool diff, bool print_ene, double t, complex **dm, complex **ddmdt, double dt, int bStart, int bEnd, string scarr, GaussianSmapling *gauss, GaussianSmapling *gauss_dot){
	if (!ionode) return;

	// open files
	bool isHole = scarr.substr(1, 4) == "hole";
	string fname;
	FILE *fil, *filtot;
	if (!diff) scarr = "_initial" + scarr;
	int idir;
	if (what == "fn"){
		fname = what + scarr + ".out"; fil = fopen(fname.c_str(), "a");
		if (diff){
			fname = what + scarr + "_tot.out";
			if (!exists(fname)){
				filtot = fopen(fname.c_str(), "a");
				fprintf(filtot, "#time(au), total f, f in each valley\n");
			}
			else filtot = fopen(fname.c_str(), "a");
		}
	}
	else if (what == "sx" || what == "sy" || what == "sz"){
		if (print_ene && !diff) error_message("print_ene && !diff is not allowed currently");
		if (what == "sx") idir = 0;
		else if (what == "sy") idir = 1;
		else if (what == "sz") idir = 2;
		if (!ddmdt && diff){
			fname = what + scarr + "_ene.out"; fil = fopen(fname.c_str(), "a");
			fname = what + scarr + "_tot.out";
			if (!exists(fname)){
				filtot = fopen(fname.c_str(), "a");
				fprintf(filtot, "#time(au), s(t), s(t) without exp(iwt)\n");
			}
			else filtot = fopen(fname.c_str(), "a");
		}
		else if (ddmdt){
			fname = "tau_" + what + scarr + "_tot.out";
			if (!exists(fname)){
				filtot = fopen(fname.c_str(), "a");
				fprintf(filtot, "#time(au), tau by FD, tau without precession, tau\n");
			}
			else filtot = fopen(fname.c_str(), "a");
		}
	}
	else if (what == "dos"){ fname = what + scarr + ".out"; fil = fopen(fname.c_str(), "a"); }

	// initialization
	double tot = 0, tot_amp = 0, dottot = 0, dottot_amp = 0, tot_tplusdt = 0, dottot_term2 = 0;
	zeros(obk, nk_glob); zeros(tot_band, nb);
	zeros(tot_valley, (int)this->latt->vpos.size()); zeros(tot_valley_band, (int)this->latt->vpos.size(), nb);
	gauss->reset();

	// compute observables
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		int iv = this->latt->whichvalley(this->elec->kvec[ik_glob]);

		for (int i = bStart; i < bEnd; i++){
			double ob = 0, ob_amp = 0, dot = 0, dot_amp = 0, ob_tplusdt = 0, dot_term2 = 0;
			if (what == "dos")
				ob = 1;
			else if (what == "fn"){
				if (diff) ob = isHole ? real(f[ik_glob][i] - dm[ik_glob][i*nb + i]) : real(dm[ik_glob][i*nb + i] - f[ik_glob][i]);
				else ob = isHole ? 1 - f[ik_glob][i] : f[ik_glob][i];
			}
			else{
				for (int b = 0; b < nb; b++){
					complex ob_kib = s[ik_glob][idir][i*nb + b];
					complex ob_kib_t = (i == b || alg.picture == "schrodinger") ? ob_kib : (ob_kib * cis((e[ik_glob][i] - e[ik_glob][b])*t));
					complex dm_kbi = (i == b) ? (dm[ik_glob][b*nb + i] - f[ik_glob][i]) : dm[ik_glob][b*nb + i];
					if (diff){
						ob += real(ob_kib_t * dm_kbi);
						ob_amp += real(ob_kib * dm_kbi);
					}
					else if (i == b) ob += real(ob_kib * f[ik_glob][i]);
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
			obk[ik_glob] += ob; tot_band[i - bStart] += ob;
			tot += ob; tot_amp += ob_amp; tot_tplusdt += ob_tplusdt;
			if (iv >= 0) { tot_valley[iv] += ob; tot_valley_band[iv][i - bStart] += ob; }
			dottot += dot; dottot_amp += dot_amp; dottot_term2 += dot_term2;

			double ene = e[ik_glob][i];
			gauss->addEvent(ene, ob);
			//gauss->addEvent2(ene, ob_amp);
		}
	}

	// total quantities
	tot *= prefac; tot_amp *= prefac; tot_tplusdt *= prefac;
	for (int i = 0; i < bEnd - bStart; i++)
		tot_band[i - bStart] *= prefac;
	for (int iv = 0; iv < this->latt->vpos.size(); iv++){
		tot_valley[iv] *= prefac;
		for (int i = 0; i < bEnd - bStart; i++)
			tot_valley_band[iv][i] *= prefac;
	}
	dottot *= prefac; dottot_amp *= prefac; dottot_term2 *= prefac;
	if (what != "dos"){
		if (!ddmdt){
			if (diff){
				if (fabs(t) < 1e-30) fprintf(filtot, " 0.00000000000000e+01 %21.14le", tot);
				else fprintf(filtot, "%21.14le %21.14le", t, tot);
				for (int iv = 0; iv < this->latt->vpos.size(); iv++)
					if (what == "fn") fprintf(filtot, " %21.14le", tot_valley[iv]);
				if (this->print_tot_band){
					for (int i = 0; i < bEnd - bStart; i++)
						fprintf(filtot, " %21.14le", tot_band[i]);
					for (int iv = 0; iv < this->latt->vpos.size(); iv++)
					for (int i = 0; i < bEnd - bStart; i++)
						if (what == "fn") fprintf(filtot, " %21.14le", tot_valley_band[iv][i]);
				}
				if (what != "fn") fprintf(filtot, " %21.14le", tot_amp);
				fprintf(filtot, "\n");
			}
			else printf("\ninitial %s %21.14le\n", what.c_str(), tot);
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
	if (what != "dos" && diff) { fflush(filtot); fclose(filtot); }

	// energy-resolved quantities
	if (!ddmdt && print_ene) fprintf(fil, "**************************************************\n");
	if (!ddmdt && print_ene) fprintf(fil, "time = %10.3e, print energy and ob(e)\n", t);
	if (!ddmdt && print_ene) fprintf(fil, "--------------------------------------------------\n");
	if (!ddmdt && print_ene) gauss->print(fil, 1.0, prefac);
	//if (!ddmdt && print_ene && (what == "sx" || what == "sy" || what == "sz")) gauss->print2(fil, 1.0, prefac);
	double dens, au2cm = 5.291772109038e-9;
	if (this->dim == 3) dens = tot / (this->VAL*std::pow(au2cm, 3));
	else if (this->dim == 2) dens = tot / (this->VAL*std::pow(au2cm, 2));
	else if (this->dim == 1) dens = tot / (this->VAL*au2cm);
	else dens = tot;
	if (!ddmdt && print_ene && what != "dos"){
		if (fabs(t) < 1e-30) fprintf(fil, "time and density:  0.00000000000000e+01 %21.14le\n", dens);
		else fprintf(fil, "time and density:  %21.14le %21.14le\n", t, dens); // carrier density in fn_ene.out
	}
	if (!ddmdt && print_ene) fprintf(fil, "**************************************************\n");
	if (what == "fn" || what == "dos" || (!ddmdt &&  diff)) fflush(fil); fflush(stdout);
	if (what == "fn" || what == "dos" || (!ddmdt &&  diff)) fclose(fil);
}

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::ddmk_ana_drive(int it, double t, complex **dm){
	MPI_Barrier(MPI_COMM_WORLD);
	if (ionode) ddmk_ana(it, t, dm);
	MPI_Barrier(MPI_COMM_WORLD);
}
template<class Tl, class Te>
void ob_1dmk<Tl, Te>::ddmk_ana(int it, double t, complex **dm){
	if (!ionode) return;
	string dir = "ddm_along_kpath_results/";
	complex *ddm = new complex[nb*nb];
	complex *ddm_res = new complex[nb*nb];
	complex *ddm_norm = new complex[nb*nb];
	string fname = dir + "ratio_tauneq_taufm.dat"; FILE *filratio;
	printf("debug ddmk_ana 1"); fflush(stdout);
	if (ddm_eq != nullptr && ddm_neq != nullptr) filratio = fopen(fname.c_str(), "wb");
	string fname_ek = dir + "ek.dat"; FILE *filek; bool opened_filek = false;
	printf("debug ddmk_ana 2"); fflush(stdout);
	if (!exists(fname_ek)) { filek = fopen(fname_ek.c_str(), "w"); opened_filek = true; }
	printf("debug ddmk_ana 3"); fflush(stdout);
	string fname_it = dir + "ddm." + int2str(it + this->it_start_ddm) + ".dat"; FILE *filit = fopen(fname_it.c_str(), "w");
	fprintf(filit, "#time = %14.7le ps\n", t / fs / 1000); fflush(filit);
	printf("debug ddmk_ana 4"); fflush(stdout);

	double trace_AB_0_tot = 0, trace_AB_1_tot = 0, trace_sq_res_tot = 0;
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		int ik_along_kpath = -1;
		for (int ik = 0; ik < ik_kpath.size(); ik++)
			if (ik_kpath[ik] == ik_glob) { ik_along_kpath = ik; break; }
		string str_ik = int2str(ik_along_kpath);
		//ostringstream oss; oss << ik_along_kpath;
		//string str_ik = oss.str();

		// ddm in schrodinger picture
		for (int i = 0; i < nb; i++)
		for (int b = 0; b < nb; b++)
		if (i == b) ddm[i*nb + b] = dm[ik_glob][i*nb + b] - f[ik_glob][i];
		else ddm[i*nb + b] = (alg.picture == "schrodinger") ? dm[ik_glob][i*nb + b] : dm[ik_glob][i*nb + b] * cis((e[ik_glob][b] - e[ik_glob][i])*t);
		
		// normalization
		double trace_sq_ddm = trace_square_hermite(ddm, nb);
		for (int ibb = 0; ibb < nb*nb; ibb++)
			ddm_norm[ibb] = ddm[ibb] / sqrt(trace_sq_ddm);

		if (ik_along_kpath >= 0){
			string fname = dir + "ddm_norm_k" + str_ik + ".out"; FILE *filddm_norm = fopen(fname.c_str(), "a");
			fprintf(filddm_norm, "%14.7le fs:", t / fs);
			fprintf_complex_mat(filddm_norm, ddm_norm, nb, "");
			fclose(filddm_norm);
			for (int i = bStart_eph; i < bEnd_eph; i++){
				if (opened_filek) fprintf(filek, "%14.7le ", e[ik_glob][i]);
				fprintf(filit, "%14.7le ", ddm[i*nb + i].real());
			}
			fprintf(filit, "\n");  if (opened_filek) fprintf(filek, "\n");
		}

		if (ddm_eq == nullptr || ddm_neq == nullptr) continue;
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
	printf("debug ddmk_ana 5"); fflush(stdout);
	fclose(filit); if (ddm_eq != nullptr && ddm_neq != nullptr) fclose(filratio); if (opened_filek) fclose(filek);

	printf("debug ddmk_ana 6"); fflush(stdout);
	if (ddm_eq == nullptr || ddm_neq == nullptr) return;
	fname = dir + "ddm_decomp_avg.out"; FILE *fildecompavg = fopen(fname.c_str(), "a");
	fprintf(fildecompavg, "%14.7le fs: %14.7le %14.7le %14.7le %14.7le\n", t / fs, sqrt(trace_sq_ddm_tot/nk_glob), 
		trace_AB_0_tot / trace_sq_ddm_tot, trace_AB_1_tot / trace_sq_ddmneq_tot, sqrt(trace_sq_res_tot/nk_glob));
	fclose(fildecompavg);
	printf("debug ddmk_ana 7"); fflush(stdout);
}