#pragma once
#include "common_headers.h"
#include "parameters.h"
#include "mymp.h"
#include "lattice.h"
#include "PumpProbe.h"
#include "electron.h"

class electronlight{
public:
	lattice* latt;
	electron* elec;
	mymp *mp;
	bool active;
	string pumpMode;
	int nk_glob, ik0_glob, ik1_glob, nk_proc, nb, bStart_dm, bEnd_dm, nb_dm, it_start; double nk_full;
	double sys_size, dt;
	double **e, **f, **e_dm, **f_dm, **fbar_dm;
	double **imEpsGS, **imEps;
	complex ***v, **v_dm, *vdag, **v_full, **pumpP, **probePpol, *probeP; // probeP0 is the energy-independent part of probeP
	complex **dm_pump, *ddmdt_contrib;

	electronlight(lattice *latt, parameters *param, electron *elec, mymp *mp)
		: latt(latt), elec(elec), mp(mp), active(fabs(pmp.pumpA0) > 1e-10), pumpMode(param->pumpMode), dt(param->tstep_pump),
		nk_glob(elec->nk), nk_full(elec->nk_full), nb(elec->nb), nb_dm(elec->nb_dm), bStart_dm(elec->bStart_dm), bEnd_dm(elec->bEnd_dm),
		e(elec->e), f(elec->f), e_dm(elec->e_dm), f_dm(elec->f_dm), v(elec->v),
		ik0_glob(mp->varstart), ik1_glob(mp->varend), nk_proc(ik1_glob - ik0_glob)
	{
		fbar_dm = alloc_real_array(nk_glob, nb_dm);
		for (int ik = 0; ik < nk_glob; ik++)
		for (int ib = 0; ib < nb_dm; ib++)
			fbar_dm[ik][ib] = 1. - f_dm[ik][ib];

		v_dm = alloc_array(3, nb_dm*nb_dm);
		pumpP = alloc_array(nk_proc, nb_dm*nb_dm); // notice that not all k points are needed for one cpu
		pumpPt = new complex[nb_dm*nb_dm]; zeros(pumpPt, nb_dm*nb_dm);
		pumpPdag = new complex[nb_dm*nb_dm]; zeros(pumpPdag, nb_dm*nb_dm);
		maux1_dm = new complex[nb_dm*nb_dm]; zeros(maux1_dm, nb_dm*nb_dm);
		maux2_dm = new complex[nb_dm*nb_dm]; zeros(maux2_dm, nb_dm*nb_dm);
		deltaRho = new complex[nb_dm*nb_dm]; zeros(deltaRho, nb_dm*nb_dm);
		ddmdt_contrib = new complex[nb_dm*nb_dm]; zeros(ddmdt_contrib, nb_dm*nb_dm);

		if (pmp.probePol.size() > 0 && pmp.probeNE > 0){
			vdag = new complex[nb*nb_dm]; zeros(vdag, nb*nb_dm);
			v_full = alloc_array(3, nb*nb);
			probePpol = alloc_array(pmp.probePol.size(), nb*nb);
			probeP = new complex[nb*nb]; zeros(probeP, nb*nb);
			probePt = new complex[nb*nb]; zeros(probePt, nb*nb);
			imEpsGS = alloc_real_array(pmp.probePol.size(), pmp.probeNE);
			imEps = alloc_real_array(pmp.probePol.size(), pmp.probeNE);
			probePdag = new complex[nb*nb]; zeros(probePdag, nb*nb);
			maux1 = new complex[nb*nb]; zeros(maux1, nb*nb);
			maux2 = new complex[nb*nb]; zeros(maux2, nb*nb);
			dm_expand = new complex[nb*nb]; zeros(dm_expand, nb*nb);
			dm1_expand = new complex[nb*nb]; zeros(dm1_expand, nb*nb);
			delta = new double[nb*nb]; zeros(delta, nb*nb);
			deltaf = new double[nb]; zeros(deltaf, nb);
		}

		compute_pumpP();

		if (pumpMode == "perturb") 
			dm_pump = alloc_array(nk_glob, nb_dm*nb_dm);

		if (ionode && pmp.probePol.size() > 0 && pmp.probeNE > 0){
			if (!param->restart){
				if (is_dir("probe_results")) system("rm -r probe_results");
				system("mkdir probe_results");
			}
			it_start = last_file_index("probe_results/imEps.",".dat");
			printf("\nit_start = %d\n", it_start); fflush(stdout);
		}
	}

	void compute_pumpP();

	complex *pumpPt, *pumpPdag, *probePt, *probePdag, *maux1, *maux2, *maux1_dm, *maux2_dm, *deltaRho, *dm_expand, *dm1_expand;
	double *deltaf, *delta;
	void pump_pert();
	inline void term_plus(double *d1, complex *m1, double *d2, complex *m2);
	inline void term_minus(complex *m1, double *d1, complex *m2, double *d2);

	bool during_pump(double t){
		if (pumpMode == "lindblad" || pumpMode == "coherent")
			return fabs(t - pmp.pump_tcenter) <= 6.1 * pmp.pumpTau;
	}
	bool enter_pump(double t, double tnext){
		return !during_pump(t) && during_pump(tnext);
	}
	bool leave_pump(double t, double tnext){
		if (pumpMode != "lindblad" && pumpMode != "coherent") return false;
		return during_pump(t) && !during_pump(tnext);
	}
	void evolve_pump(double t, complex** dm, complex** dm1, complex** ddmdt_pump);
	void evolve_pump_coh(double t, complex** dm, complex** dm1, complex** ddmdt_pump);
	void evolve_pump_lindblad(double t, complex** dm, complex** dm1, complex** ddmdt_pump);
	inline void compute_pumpPt_coh(double t, complex *Pk, double *ek);
	inline void compute_pumpPt(double t, complex *Pk, double *ek);
	inline void term_plus(complex *dm1, complex *a, complex *dm, complex *b);
	inline void term_minus(complex *a, complex *dm1, complex *b, complex *dm);

	void probe(int it, double t, complex **dm, complex **dm1);
	void calcImEps(double t, complex **dm, complex **dm1);
	inline void expand_denmat(int ik_glob, complex *dm, complex *dm1);
	inline void compute_probePt(double t, complex *Pk, double *ek);
	void write_imEpsVSomega(string fname);
	inline void term_plus_probe(complex *dm1, complex *a, complex *dm, complex *b);
	inline void term_minus_probe(complex *a, complex *dm1, complex *b, complex *dm);
};