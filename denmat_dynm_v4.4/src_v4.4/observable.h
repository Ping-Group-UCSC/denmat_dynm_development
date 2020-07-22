#pragma once
#include "common_headers.h"
#include "parameters.h"

template<class Tl, class Te>
class observable{
public:
	Tl* latt;
	Te* elec;
	int dim, freq_measure_ene;
	double VAL;
	observable(Tl* latt, parameters* param, Te* elec)
		:latt(latt), elec(elec), dim(latt->dim), freq_measure_ene(param->freq_measure_ene)
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
	int nk_glob, nb, nv; // notice that n is nbs
	double prefac;
	FILE *fil, *filtot;
	double **e, **f;
	double emin, emax, evmax, ecmin;
	complex ***s;
	GaussianSmapling *gauss_elec, *gauss_hole;

	ob_1dmk(Tl* latt, parameters *param, Te* elec)
		:observable<Tl, Te>(latt, param, elec), 
		nk_glob(elec->nk), nb(elec->nb_dm), nv(elec->nv_dm), prefac(1. / elec->nk_full), e(elec->e_dm), f(elec->f_dm)
	{
		if (nb > nv){
			emax = maxval(e, nk_glob, nv, nb);
			ecmin = minval(e, nk_glob, nv, nb);
			if (ionode) printf("emax = %lg ecmin = %lg\n", emax, ecmin);
			gauss_elec = new GaussianSmapling(ecmin, param->de_measure, emax, param->degauss_measure);
		}
		if (nv > 0){
			evmax = maxval(e, nk_glob, 0, nv);
			emin = minval(e, nk_glob, 0, nv);
			if (ionode) printf("evmax = %lg emin = %lg\n", evmax, emin);
			gauss_hole = new GaussianSmapling(emin, param->de_measure, evmax, param->degauss_measure);
		}
		this->s = elec->s;
	}

	void measure(string what, bool isDot, bool diff, bool print_ene, double t, complex **dm);
	void measure_brange(string what, bool isDot, bool diff, bool print_ene, double t, complex **dm, int bStart, int bEnd, string scarr, GaussianSmapling *gauss);
};

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::measure(string what, bool isDot, bool diff, bool print_ene, double t, complex **dm){ // if isDot, dm will mean ddmdt
	if (isDot && print_ene) error_message("isDot && print_ene");
	if (nb > nv)
		measure_brange(what, isDot, diff, print_ene, t, dm, nv, nb, "elec", gauss_elec);
	if (nv > 0)
		measure_brange(what, isDot, diff, print_ene, t, dm, 0, nv, "hole", gauss_hole);
}

template<class Tl, class Te>
void ob_1dmk<Tl, Te>::measure_brange(string what, bool isDot, bool diff, bool print_ene, double t, complex **dm, int bStart, int bEnd, string scarr, GaussianSmapling *gauss){
	if (alg.picture != "interaction")
		error_message("alg_picture must be interaction now");
	MPI_Barrier(MPI_COMM_WORLD);
	if (!ionode) return;

	bool b_tot = true;
	string fname;
	if (!diff) scarr = "initial_" + scarr;
	if (isDot) scarr = "dot_" + scarr;
	int idir;
	double sign_isy = 0;
	if (what == "sx"){
		idir = 0;
		fname = "Sx_" + scarr + "_ene.out"; fil = fopen(fname.c_str(), "a");
		fname = "Sx_" + scarr + "_tot.out";  filtot = fopen(fname.c_str(), "a");
	}
	if (what == "sy"){
		idir = 1;
		fname = "Sy_" + scarr + "_ene.out"; fil = fopen(fname.c_str(), "a");
		fname = "Sy_" + scarr + "_tot.out";  filtot = fopen(fname.c_str(), "a");
	}
	if (what == "sz"){
		idir = 2;
		fname = "Sz_" + scarr + "_ene.out"; fil = fopen(fname.c_str(), "a");
		fname = "Sz_" + scarr + "_tot.out";  filtot = fopen(fname.c_str(), "a");
	}
	if (what == "dos"){
		fname = "dos_" + scarr + ".out"; fil = fopen(fname.c_str(), "a");
		b_tot = false;
	}
	if (what == "fn"){
		fname = "fn_" + scarr + ".out"; fil = fopen(fname.c_str(), "a");
		b_tot = false;
	}
	if (isDot && !b_tot) error_message("isDot && !b_tot");

	double tot = 0, tot_amp = 0;
	gauss->reset();

	if (print_ene) fprintf(fil, "**************************************************\n");
	if (print_ene) fprintf(fil, "time = %10.3e, print energy and ob(e)\n", t);
	if (print_ene) fprintf(fil, "--------------------------------------------------\n");

	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		//bool isK = this->latt->isKvalley(this->elec->kvec[ik_glob]);
		//if ((what == "sxk" || what == "syk") && (!isK)) continue;

		for (int i = bStart; i < bEnd; i++){
			double obtmp = 0, obtmp_amp = 0;
			if (what == "dos")
				obtmp = 1;
			else if (what == "fn"){
				if (diff)
					obtmp = real(dm[ik_glob][i*nb + i] - f[ik_glob][i]);
				else
					obtmp = f[ik_glob][i];
			}
			else{
				for (int b = bStart; b < bEnd; b++){
					complex ob_kib = s[ik_glob][idir][i*nb + b] + complex(0, sign_isy) * s[ik_glob][1][i*nb + b];
					if (isDot){
						obtmp += real(ob_kib * cis((e[ik_glob][i] - e[ik_glob][b])*t) * dm[ik_glob][b*nb + i]);
						obtmp_amp += real(ob_kib * dm[ik_glob][b*nb + i]);
					}
					else{
						if (diff){
							if (i == b){
								obtmp += real(ob_kib * (dm[ik_glob][b*nb + i] - f[ik_glob][i]));
								obtmp_amp += real(ob_kib * (dm[ik_glob][b*nb + i] - f[ik_glob][i]));
							}
							else{
								obtmp += real(ob_kib * cis((e[ik_glob][i] - e[ik_glob][b])*t) * dm[ik_glob][b*nb + i]);
								obtmp_amp += real(ob_kib * dm[ik_glob][b*nb + i]);
							}
						}
						else{
							if (i == b)
								obtmp += real(ob_kib * f[ik_glob][i]);
						}
					}
				}
			}
			tot += obtmp; tot_amp += obtmp_amp;

			double ene = e[ik_glob][i];
			gauss->addEvent(ene, obtmp);
			gauss->addEvent2(ene, obtmp_amp);
		}
	}

	tot *= prefac; tot_amp *= prefac;
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
		if (fabs(t) < 1e-30)
			fprintf(filtot, " 0.0000000000e+01 %21.14le %21.14le\n", tot, tot_amp);
		else
			fprintf(filtot, "%21.14le %21.14le %21.14le\n", t, tot, tot_amp);
	}
	if (b_tot) fflush(filtot);
	if (b_tot) fclose(filtot);
	if (print_ene && !b_tot) gauss->print(fil, 1.0, prefac);
	if (print_ene && b_tot) gauss->print2(fil, 1.0, prefac);
	if (print_ene && !b_tot)
	if (fabs(t) < 1e-30)
		fprintf(fil, "time and density:  0.0000000000e+01 %21.14le\n", dens);
	else
		fprintf(fil, "time and density:  %21.14le %21.14le\n", t, dens); // carrier density in fn_ene.out
	if (print_ene) fprintf(fil, "**************************************************\n");
	fflush(fil); fflush(stdout);
	fclose(fil);
}
