// assume the integral of denmat(t) is not needed, so that only denmat at one time (or alternatively a few times) is needed
#pragma once
#include "common_headers.h"
#include "parameters.h"
#include "mymp.h"

class denmat{
public:
	FILE *fil;
	int nt; // number of time steps
	double t, t0, tend, dt, dt_pump; // atomic units, 40 is about 1fs
	denmat(parameters *param)
		:t0(param->t0), tend(param->tend), dt(param->tstep), dt_pump(param->tstep_pump), t(param->t0){}
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
	int nk_glob, ik0_glob, ik1_glob, nk_proc, nb;
	complex **dm, **oneminusdm, **ddmdt, **ddmdt_term, **dm_eq;
	double mue, muh, **f_eq;

	singdenmat_k(parameters *param, mymp *mp, int nk_glob, int nb)
		:denmat(param), mp(mp), 
		nk_glob(nk_glob), ik0_glob(mp->varstart), ik1_glob(mp->varend), nk_proc(ik1_glob - ik0_glob), nb(nb),
		mue(param->mu), muh(param->mu)
	{
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

	void set_dm_eq(double t, double **e, int nv);
	void set_dm_eq(bool isHole, double t, double mu0, double **e, int bStart, int bEnd);
	double find_mu(bool isHole, double ncarrier, double t, double mu0, double **e, int bStart, int bEnd);
	double compute_ncarrier_eq(bool isHole, double t, double mu, double **e, int bStart, int bEnd);

	void write_ddmdt();
	void write_dm();
	void write_dm_tofile(double t);

	// coherent dynamics
	complex **Hcoh, *Hcoht;
	double **e;
	complex prefac_coh;
	void init_Hcoh(complex **H_BS, double **e);
	void compute_Hcoht(double t, complex *H, double *e);
	void evolve_coh(double t, complex** ddmdt_coh);
};
