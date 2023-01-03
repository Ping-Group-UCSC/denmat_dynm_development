#pragma once
#include "common_headers.h"
#include "parameters.h"

class phenom_relax{
// rhodot = - (rho - rho_eq) / tau
public:
	double tau;
	int nk_glob, bStart, bEnd, nb, nb_dm; // bStart and bEnd relative to bStart_dm
	//double **e;
	complex **dm_eq, **dm, **ddmdt_phenom;

	//phenom_relax(parameters *param, int nk_glob, int nb_dm, complex **dm_eq_extend, double **e_extend)
	phenom_relax(parameters *param, int nk_glob, int nb_dm, complex **dm_eq_extend)
		: tau(param->tau_phenom), bStart(param->bStart_tau), bEnd(param->bEnd_tau), nb(bEnd - bStart),
		nk_glob(nk_glob), nb_dm(nb_dm)
	{
		if (bEnd > nb_dm)
			error_message("bEnd_tau > nb_dm", "phenom_relax");
		if (ionode) printf("\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) { printf("initialize phenom_relax for bStart = %d and bEnd = %d\n", bStart, bEnd); fflush(stdout); }
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");
		dm_eq = alloc_array(nk_glob, nb*nb);
		dm = alloc_array(nk_glob, nb*nb);
		ddmdt_phenom = alloc_array(nk_glob, nb*nb);
	}

	void evolve_driver(double t, complex **dm_extend, complex **dm_eq_extend, complex **ddmdt_phenom_extend);
	void evolve(double t, complex **dm, complex **dm_eq, complex **ddmdt_phenom);
};

void phenom_relax::evolve_driver(double t, complex **dm_extend, complex **dm_eq_extend, complex **ddmdt_phenom_extend){
	trunc_copy_arraymat(dm, dm_extend, nk_glob, nb_dm, bStart, bEnd);
	trunc_copy_arraymat(dm_eq, dm_eq_extend, nk_glob, nb_dm, bStart, bEnd);
	zeros(ddmdt_phenom, nk_glob, nb*nb);

	evolve(t, dm, dm_eq, ddmdt_phenom);

	zeros(ddmdt_phenom_extend, nk_glob, nb_dm*nb_dm);
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
		set_mat(ddmdt_phenom_extend[ik_glob], ddmdt_phenom[ik_glob], nb_dm, bStart, bEnd, bStart, bEnd);
}

void phenom_relax::evolve(double t, complex **dm, complex **dm_eq, complex **ddmdt_phenom){
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++)
		ddmdt_phenom[ik_glob][i*nb + j] += -(dm[ik_glob][i*nb + j] - dm_eq[ik_glob][i*nb + j]) / tau;
}