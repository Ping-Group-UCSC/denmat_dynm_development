#pragma once
#include "common_headers.h"
#include "lattice.h"
#include "parameters.h"
#include "electron.h"
#include "phonon.h"

class electronphonon{
public:
	bool sepr_eh, isHole; // when code=="jdftx", input for holes and electrons are different
	int bStart, bEnd, nb_expand; // bStart and bEnd relative to bStart_dm; nb_expand = nb_dm
	lattice *latt;
	electron *elec;
	phonon *ph;
	const double t0, tend, degauss;
	const double prefac_gaussexp, prefac_sqrtgaussexp;

	double prefac_gauss, prefac_sqrtgauss, prefac_eph;
	int nk_glob, nk_proc, ik0_glob, ik1_glob;
	int nkpair_glob, nkpair_proc, ikpair0_glob, ikpair1_glob; // kp means kpair
	size_t *k1st, *k2nd; // use size_t to be consistent with jdftx
	int nm, nb;
	complex ***App, ***Amm, ***Apm, ***Amp; // App=Gp*sqrt(nq+1), Amm=Gm*sqrt(nq), Apm=Gp*sqrt(nq), Amp=Gm*sqrt(nq+1)
	complex **P1, **P2, *P1_next, *P2_next;
	sparse2D *sP1, *sP2;
	sparse_mat *sm1_next, *sm2_next;
	int *ij2i, *ij2j;

	electronphonon(parameters *param, bool sepr_eh = false, bool isHole = false)
		:sepr_eh(false), isHole(false), degauss(param->degauss), prefac_gaussexp(-0.5 / std::pow(param->degauss, 2)),
		prefac_sqrtgaussexp(-0.25 / std::pow(param->degauss, 2)),
		prefac_gauss(1. / (sqrt(2 * M_PI) * param->degauss)), prefac_sqrtgauss(1. / sqrt(sqrt(2 * M_PI) * param->degauss)),
		t0(param->t0), tend(param->tend)
	{}
	electronphonon(lattice *latt, parameters *param, bool sepr_eh = false, bool isHole = false)
		:latt(latt), sepr_eh(false), isHole(false), degauss(param->degauss), prefac_gaussexp(-0.5 / std::pow(param->degauss, 2)),
		prefac_sqrtgaussexp(-0.25 / std::pow(param->degauss, 2)),
		prefac_gauss(1. / (sqrt(2 * M_PI) * param->degauss)), prefac_sqrtgauss(1. / sqrt(sqrt(2 * M_PI) * param->degauss)),
		t0(param->t0), tend(param->tend)
	{}
	electronphonon(lattice *latt, parameters *param, electron *elec, phonon *ph, bool sepr_eh = false, bool isHole = false)
		:latt(latt), sepr_eh(sepr_eh), isHole(isHole), elec(elec), nk_glob(elec->nk), ph(ph), nm(ph->nm),
		degauss(param->degauss), prefac_gaussexp(-0.5 / std::pow(param->degauss, 2)),
		prefac_sqrtgaussexp(-0.25 / std::pow(param->degauss, 2)),
		prefac_gauss(1. / (sqrt(2 * M_PI) * param->degauss)), prefac_sqrtgauss(1. / sqrt(sqrt(2 * M_PI) * param->degauss)),
		t0(param->t0), tend(param->tend),
		prefac_eph(2 * M_PI / elec->nk_full)
	{
		nb_expand = elec->nb_dm;
		get_brange(sepr_eh, isHole);

		alloc_nonparallel();

		if (alg.expt)
			this->e = trunc_alloccopy_array(elec->e_dm, nk_glob, bStart, bEnd);

		get_nkpair();
	}

	void alloc_nonparallel(){
		ddmdt_contrib = new complex[nb*nb];
		maux1 = new complex[nb*nb];
		maux2 = new complex[nb*nb];
		if (alg.ddmeq)
			ddm_eq = alloc_array(nk_glob, nb*nb);
	}

	inline double gauss_exp(double e){
		return exp(prefac_gaussexp * std::pow(e,2));
	}
	inline double sqrt_gauss_exp(double e){
		return exp(prefac_sqrtgaussexp * std::pow(e, 2));
	}

	mymp *mp;
	void set_eph(mymp *mp);
	void get_brange(bool sepr_eh, bool isHole);
	void get_nkpair();
	void alloc_ephmat(int, int);
	void set_kpair();
	void read_ldbd_kpair();
	void set_ephmat();
	void read_ldbd_eph();
	void make_map();
	void set_sparseP();

	// evolve
	complex **ddm_eq;
	complex *ddmdt_contrib, *maux1, *maux2, *P1t, *P2t;
	sparse_mat *smat1_time, *smat2_time;
	double **e;
	complex **dm, **dm1, **ddmdt_eph;

	void compute_ddm_eq(double **f0_expand);
	void evolve_driver(double t, complex **dm_expand, complex **dm1_expand, complex **ddm_expand);
	void evolve(double t, complex **dm, complex **dm1, complex **ddm);
	void compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, complex *p1, complex *p2, complex *ddmk);
	void compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, sparse_mat *sm1, sparse_mat *sm2, complex *ddmk);
	void compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, complex **App, complex **Amm, complex **Apm, complex **Amp, complex *ddmk);
	inline void term1_P(complex *dm1, complex *p, complex *dm);
	inline void term2_P(complex *dm1, complex *p, complex *dm);
	inline void term1_sP(complex *dm1, sparse_mat *sm, complex *dm);
	inline void term2_sP(complex *dm1, sparse_mat *sm, complex *dm);
	inline void term1(complex *dm1, complex *a, complex *dm, complex *b);
	inline void term1(FILE *fp, complex *dm1, complex *a, complex *dm, complex *b);
	inline void term2(complex *a, complex *dm1, complex *b, complex *dm);
	inline void term2(FILE *fp, complex *a, complex *dm1, complex *b, complex *dm);
	inline void hermite(complex *m, complex *h){
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			h[i*nb + j] = conj(m[j*nb + i]);
	}
	inline void compute_Pt(double t, complex *P1, complex *P2, double *ek, double *ekp);
	inline void init_sparse_mat(sparse_mat *sin, sparse_mat *sout);
	inline void compute_sPt(double t, sparse_mat *sm1, sparse_mat *sm2, double *ek, double *ekp);
};