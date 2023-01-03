// assume the integral of denmat(t) is not needed, so that only denmat at one time (or alternatively a few times) is needed
#pragma once
#include "common_headers.h"
#include "parameters.h"
#include "electron.h"
#include "mymp.h"

class denmat{
public:
	FILE *fil;
	int nt; // number of time steps
	double t, t0, tend, dt, dt_laser; // atomic units, 40 is about 1fs
	denmat(parameters *param)
		:t0(param->t0), tend(param->tend), dt(param->tstep), dt_laser(param->tstep_laser), t(param->t0){}
};

// not used now!!! size (nk*nb)x(nk*nb)
class singdenmat:public denmat{
public:
	int n;
	//complex *dm;
	singdenmat(parameters *param, int n) :denmat(param), n(n){
		//dm = new complex[n*n];
	}
};

// use array instead of 2D matrix for cblas
// size nk (nb)x(nb)
class singdenmat_k :public denmat{
public:
	mymp *mp;
	electron *elec;
	int nk_glob, ik0_glob, ik1_glob, nk_proc, nb;
	complex **dm, **oneminusdm, **ddmdt, **ddmdt_term, **dm_eq;
	double mue, muh, **f_eq, ne, nh;

	singdenmat_k(parameters *param, mymp *mp, electron *elec)
		:denmat(param), mp(mp), elec(elec),
		nk_glob(elec->nk), ik0_glob(mp->varstart), ik1_glob(mp->varend), nk_proc(ik1_glob - ik0_glob), nb(elec->nb_dm),
		mue(param->mu), muh(param->mu)
	{
		if (ionode) printf("\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("k-diagonal single-particle density matrix\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");

		dm = alloc_array(nk_glob, nb*nb);
		oneminusdm = alloc_array(nk_glob, nb*nb);
		ddmdt = alloc_array(nk_glob, nb*nb);
		ddmdt_term = alloc_array(nk_glob, nb*nb);
		dm_eq = alloc_array(nk_glob, nb*nb);
		f_eq = alloc_real_array(nk_glob, nb);
	}

	void init_dm(double **f);
	void init_dm(complex **dm0);
	void read_dm_restart();
	void read_ldbd_dm0();
	void update_ddmdt(complex **ddmdt_term);
	void set_oneminusdm();
	void update_dm_euler(double dt);

	complex **ddm_eq, **ddm_neq;
	double *trace_sq_ddm_eq, *trace_sq_ddm_neq;
	//void init_dmDP(complex **ddm_eq, complex **ddm_neq);
	//void use_dmDP(double **f);

	void set_dm_eq(double t, double **e, int nv);
	void set_dm_eq(bool isHole, double t, double mu0, double **e, int bStart, int bEnd);
	double find_mu(bool isHole, double carrier_bvk, double temperature, double mu0, double **e, int bStart, int bEnd);
	double compute_nfree_eq(bool isHole, double temperature, double mu, double **e, int bStart, int bEnd);
	void compute_f(double temperature, double mue, double muh, double **e, int nv){
		zeros(f_eq, nk_glob, nb);
		for (int ik = ik0_glob; ik < ik1_glob; ik++){
			for (int i = 0; i < nv; i++)
				f_eq[ik][i] = electron::fermi(temperature, muh, e[ik][i]);
			for (int i = nv; i < nb; i++)
				f_eq[ik][i] = electron::fermi(temperature, mue, e[ik][i]);
		}
		mp->allreduce(f_eq, nk_glob, nb, MPI_SUM);
	}

	void write_ddmdt(std::vector<vector3<double>> kvec, std::vector<int> ik_kpath, double **e);
	void write_dm();
	void write_dm_tofile(double t);

	// coherent dynamics
	complex **Hcoh, *Hcoht;
	double **e;
	complex prefac_coh;
	void init_Hcoh(complex **H_BS, complex **H_Ez, double **e);
	void compute_Hcoht(double t, complex *H, double *e);
	void evolve_coh(double t, complex** ddmdt_coh);
};
