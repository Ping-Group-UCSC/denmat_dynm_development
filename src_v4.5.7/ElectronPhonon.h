#pragma once
#include "common_headers.h"
#include "lattice.h"
#include "parameters.h"
#include "electron.h"
#include "phonon.h"
#include "ElectronImpurity.h"
#include "ElecImp_Model.h"
#include "ElecElec_Model.h"

// e-ph here also contains e-i and e-e, so in future we need to reorganise the source codes related to the scattering

class electronphonon{
public:
	mymp *mp;
	bool sepr_eh, isHole; // when code=="jdftx", input for holes and electrons are different
	int bStart, bEnd, nb_expand; // bStart and bEnd relative to bStart_dm; nb_expand = nb_dm
	double eStart, eEnd;
	lattice *latt;
	electron *elec;
	phonon *ph;
	coulomb_model *coul_model;
	electronimpurity **eimp;
	elecelec_model *ee_model;
	const double t0, tend, degauss;
	const double prefac_gaussexp, prefac_sqrtgaussexp;

	double prefac_gauss, prefac_sqrtgauss, prefac_eph, scale_scatt, scale_eph, scale_ei, scale_ee;
	int nk_glob, nk_proc, ik0_glob, ik1_glob;
	int nkpair_glob, nkpair_proc, ikpair0_glob, ikpair1_glob; // kp means kpair
	size_t *k1st, *k2nd; // use size_t to be consistent with jdftx
	int nm, nb, nv, nc;
	complex ***App, ***Amm, ***Apm, ***Amp; // App=Gp*sqrt(nq+1), Amm=Gm*sqrt(nq), Apm=Gp*sqrt(nq), Amp=Gm*sqrt(nq+1)
	complex **P1, **P2, *P1_next, *P2_next, **dP1ee, **dP2ee;
	sparse2D *sP1, *sP2;
	sparse_mat *sm1_next, *sm2_next;
	int *ij2i, *ij2j;

	electronphonon(parameters *param, bool sepr_eh = false, bool isHole = false)
		:sepr_eh(false), isHole(false), degauss(param->degauss), prefac_gaussexp(-0.5 / std::pow(param->degauss, 2)),
		prefac_sqrtgaussexp(-0.25 / std::pow(param->degauss, 2)),
		prefac_gauss(1. / (sqrt(2 * M_PI) * param->degauss)), prefac_sqrtgauss(1. / sqrt(sqrt(2 * M_PI) * param->degauss)),
		scale_scatt(param->scale_scatt), scale_eph(param->scale_eph), scale_ei(param->scale_ei), scale_ee(param->scale_ee),
		t0(param->t0), tend(param->tend)
	{}
	electronphonon(lattice *latt, parameters *param, bool sepr_eh = false, bool isHole = false)
		:latt(latt), sepr_eh(false), isHole(false), degauss(param->degauss), prefac_gaussexp(-0.5 / std::pow(param->degauss, 2)),
		prefac_sqrtgaussexp(-0.25 / std::pow(param->degauss, 2)),
		prefac_gauss(1. / (sqrt(2 * M_PI) * param->degauss)), prefac_sqrtgauss(1. / sqrt(sqrt(2 * M_PI) * param->degauss)),
		scale_scatt(param->scale_scatt), scale_eph(param->scale_eph), scale_ei(param->scale_ei), scale_ee(param->scale_ee),
		t0(param->t0), tend(param->tend)
	{}
	electronphonon(mymp *mp, lattice *latt, parameters *param, electron *elec, phonon *ph, bool sepr_eh = false, bool isHole = false)
		:mp(mp), latt(latt), sepr_eh(sepr_eh), isHole(isHole), elec(elec), nk_glob(elec->nk), ph(ph), nm(ph->nm),
		degauss(param->degauss), prefac_gaussexp(-0.5 / std::pow(param->degauss, 2)),
		prefac_sqrtgaussexp(-0.25 / std::pow(param->degauss, 2)),
		prefac_gauss(1. / (sqrt(2 * M_PI) * param->degauss)), prefac_sqrtgauss(1. / sqrt(sqrt(2 * M_PI) * param->degauss)),
		scale_scatt(param->scale_scatt), scale_eph(param->scale_eph), scale_ei(param->scale_ei), scale_ee(param->scale_ee),
		t0(param->t0), tend(param->tend),
		need_imsig(param->need_imsig),
		prefac_eph(2 * M_PI / elec->nk_full),
		coul_model(nullptr), eimp(nullptr), f_eq(nullptr), ee_model(nullptr), sP1(nullptr), sP2(nullptr),
		dP1ee(nullptr), dP2ee(nullptr)
	{
		if (ionode) printf("\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("electron scattering: e-ph, e-i, e-e\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");

		if (!alg.scatt_enable) return;
		nb_expand = elec->nb_dm;
		get_brange(sepr_eh, isHole);

		alloc_nonparallel();

		//if (alg.expt)
		this->e = trunc_alloccopy_array(elec->e_dm, nk_glob, bStart, bEnd);

		get_nkpair();

		if (clp.scrMode != "none")
			coul_model = new coulomb_model(latt, param, elec, bStart, bEnd, eEnd - eStart);
		if (eip.ni.size() > 0) eimp = new electronimpurity*[eip.ni.size()]{nullptr};
		for (int iD = 0; iD < eip.ni.size(); iD++){
			eimp[iD] = new electronimpurity(iD, mp, isHole, nkpair_glob, nb, latt->cell_size);
			if (eip.impMode[iD] == "model_ionized")
				eimp[iD]->eimp_model = new elecimp_model(iD, latt, param, elec, bStart, bEnd, eStart, eEnd, coul_model);
		}
		if (eep.eeMode != "none" && alg.summode)
			ee_model = new elecelec_model(latt, param, elec, bStart, bEnd, eStart, eEnd, coul_model);
	}

	inline double gauss_exp(double e){
		return exp(prefac_gaussexp * std::pow(e, 2));
	}
	inline double sqrt_gauss_exp(double e){
		return exp(prefac_sqrtgaussexp * std::pow(e, 2));
	}

	// Setups
	void alloc_nonparallel(){
		ddmdt_contrib = new complex[nb*nb];
		maux1 = new complex[nb*nb];
		maux2 = new complex[nb*nb];
		if (alg.ddmdteq) ddmdt_eq = alloc_array(nk_glob, nb*nb);
		if (alg.linearize || alg.linearize_dPee) f_eq = alloc_real_array(nk_glob, nb);
		if (alg.linearize_dPee) f1_eq = alloc_real_array(nk_glob, nb);
	}
	
	void set_eph();
	void get_brange(bool sepr_eh, bool isHole);
	void get_nkpair();
	void alloc_ephmat(int, int);
	void set_kpair();
	void read_ldbd_kpair();
	void set_ephmat();
	void read_ldbd_eph();
	void make_map();
	void set_sparseP(bool fisrt_call);
	void add_scatt_contrib(string what, int iD=0, complex **dm = nullptr, complex **dm1 = nullptr, double t = 0);
	//void reset_scatt(bool reset_eimp, bool reset_ee, double nfree, complex **dm, complex **dm1, double t);
	void reset_scatt(bool reset_eimp, bool reset_ee, complex **dm, complex **dm1, double t, double **f_eq_expand = nullptr);

	// Analysis
	bool need_imsig;
	void compute_imsig();
	void compute_imsig(string what);
	matrix3<> compute_conductivity_brange(double **imsigp, double ***v, double **f, int bStart, int bEnd);//return conductivity
	matrix3<> compute_conductivity_brange(double ***dfdEfield, double ***v, int bStart, int bEnd);//return conductivity
	matrix3<> compute_mobility_brange(matrix3<> cond, double **f, int bStart, int bEnd, string scarr, bool print = true);//return conductivity
	void write_conductivity(matrix3<> cond);
	void analyse_g2(double de, double degauss, double degthr);
	void analyse_g2_ei(double de, double degauss, double degthr);

	// Linearize the scattering term of the density-matrix master equation
	double **f_eq, **f1_eq; // f1_eq = 1 - f_eq
	complex **Lscij, **Lscji, *Lsct, **Lscii; // Linear operator of the scattering term of master equation. "ij" for ki <= kj; "ji" for kj <= ki;
		// ii for parts: - \sum_3 [ (1-f3) conj(P_331a) delta_2b + delta_1a P_b233 f3 ] where k1=k2=ka=kb
	sparse2D *sLscij, *sLscji;
	void set_Lsc(double **f_eq);
	void set_Leph_from_jdftx_data();
	void add_Lsc_contrib(string what, int iD=0);
	void add_Lscii_from_P1_P2(complex *P1, complex *P2, int ik, int jk);
	void add_Lscij_from_P1_P2(complex *P1, complex *P2, int ikpair_local, int ik, int jk, bool fac2 = false);
	void add_Lscij_from_P3_P4(complex *P3, complex *P4, int ikpair_local, int ik, int jk);
	void add_Lscij_from_P5_P6(complex *P5, complex *P6, int ikpair_local, int ik, int jk);

	// evolve
	complex **ddmdt_eq;
	complex *ddmdt_contrib, *maux1, *maux2, *P1t, *P2t;
	sparse_mat *smat1_time, *smat2_time;
	double **e;
	complex **dm, **dm1, **ddmdt_eph;

	void compute_ddmdt_eq(double **f0_expand);
	void evolve_driver(double t, complex **dm_expand, complex **dm1_expand, complex **ddmdt_expand, bool compute_eq = false);
	void evolve(double t, complex **dm, complex **dm1, complex **ddmdt, bool compute_eq = false);

	// linearization
	void evolve_linear(double t, complex **dm, complex **ddmdt);
	void compute_ddmdt(complex *dmkp, complex *lsc, complex *ddmdtk);

	inline void compute_Pt(double t, double *ek, double *ekp, complex *P, complex *Pt, bool minus);
	inline void init_sparse_mat(sparse_mat *sin, sparse_mat *sout, bool copy_elem = false);
	inline void compute_sPt(double t, double *ek, double *ekp, sparse_mat *sm, sparse_mat *smt, bool minus);
};