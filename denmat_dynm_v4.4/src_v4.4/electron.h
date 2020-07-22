#pragma once
#include "common_headers.h"
#include "lattice.h"
#include "parameters.h"
#include "PumpProbe.h"

class electron{
public:
	lattice *latt;
	double temperature, mu;
	int nk1, nk2, nk3, nk_full;
	int ns, nk, nb; // spin, k point, band
	int nv, nc; // valence and conduction bands
	int bStart_dm, bEnd_dm, nb_dm, nv_dm, nc_dm; // band range related to density matrix
	std::vector<vector3<>> kvec;
	double **e, **f, **e_dm, **f_dm; // ocuupation number
	vector3<> B;
	complex ***s, ***v;
	complex **dm_Bpert;
	double degthr;
	complex **H_BS;

	electron(parameters *param)
		:temperature(param->temperature), mu(param->mu),
		nk1(param->nk1), nk2(param->nk2), nk3(param->nk3), nk_full(param->nk1*param->nk2*param->nk3), B(param->B){}
	electron(lattice *latt, parameters *param)
		:latt(latt), temperature(param->temperature), mu(param->mu), 
		nk1(param->nk1), nk2(param->nk2), nk3(param->nk3), nk_full(param->nk1*param->nk2*param->nk3), B(param->B)
	{
		if (code == "jdftx"){
			read_ldbd_size();
			alloc_mat();
			read_ldbd_kvec();
			read_ldbd_ek();
			e_dm = trunc_alloccopy_array(e, nk, bStart_dm, bEnd_dm);
			f_dm = trunc_alloccopy_array(f, nk, bStart_dm, bEnd_dm);
			read_ldbd_smat();
			if (pmp.pumpA0 > 0) read_ldbd_vmat();

			compute_dm_Bpert_1st(param->Bpert, param->t0);
		}
	}
	void alloc_mat();

	vector3<> get_kvec(int&, int&, int&);
	static double fermi(double t, double mu, double e){
		double ebyt = (e - mu) / t;
		if (ebyt < -46) return 1;
		else if (ebyt > 46) return 0;
		else return 1. / (exp(ebyt) + 1);
	}

	void read_ldbd_size();
	void read_ldbd_kvec();
	void read_ldbd_ek();
	void read_ldbd_smat();
	void read_ldbd_vmat();

	void compute_dm_Bpert_1st(vector3<> Bpert, double t0);

	void set_H_BS(int ik0_glob, int ik1_glob);
};