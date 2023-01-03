#include "electron.h"
/*
void electron::compute_DP_related(vector3<> Bpert){
	if (dm_Bpert == nullptr) error_message("compute_DP_related requires matrix dm_pert");
	//if (alg.picture != "schrodinger") error_message("compute_DP_related works in schrodinger picture");
	MPI_Barrier(MPI_COMM_WORLD);
	if (ionode) printf("\n**************************************************\n");
	if (ionode) printf("Compute DP-related quatities:\n");

	ddm_Bpert = trunc_alloccopy_arraymat(dm_Bpert, nk, nb_dm, 0, nb_dm);
	complex **ddm_Bpert_off = trunc_alloccopy_arraymat(dm_Bpert, nk, nb_dm, 0, nb_dm);
	double *trace_sq_ddm_Bpert = new double[nk];
	double *trace_sq_ddm_Bpert_off = new double[nk];
	has_precess = new bool[nk];
	for (int ik = 0; ik < nk; ik++){
		for (int i = 0; i < nb_dm; i++){
			ddm_Bpert[ik][i*nb_dm + i] -= f_dm[ik][i];
			ddm_Bpert_off[ik][i*nb_dm + i] = c0;
		}
		trace_sq_ddm_Bpert[ik] = trace_square_hermite(ddm_Bpert[ik], nb_dm);
		trace_sq_ddm_Bpert_off[ik] = trace_square_hermite(ddm_Bpert_off[ik], nb_dm);
		has_precess[ik] = sqrt(trace_sq_ddm_Bpert_off[ik]) / sqrt(trace_sq_ddm_Bpert[ik]) > 1e-8;
	}
	ddm_Bpert_neq = alloc_array(nk, nb_dm*nb_dm);
	if (ionode) printf("Tr[ddm_eq ddm_eq] = %lg\n", trace_square_hermite(ddm_Bpert, nk, nb_dm) / nk);
	dm_Bpert_neq = alloc_array(nk, nb_dm*nb_dm);
	complex **double_commut = alloc_array(nk, nb_dm*nb_dm);
	vector3<double> Bnorm = normalize(Bpert);
	complex **dBS = alloc_array(nk, nb_dm*nb_dm);
	DP_precess_fac = new double[nk];
	double dstot = 0;

	for (int ik = 0; ik < nk; ik++){
		commutator_mat_diag(ddm_Bpert_neq[ik], e_dm[ik], ddm_Bpert[ik], nb_dm, cmi); // [ek,ddm_eq_k]
		if (has_precess[ik]){
			double trace_sq = trace_square_hermite(ddm_Bpert_neq[ik], nb_dm);
			has_precess[ik] = sqrt(trace_sq) / sqrt(trace_sq_ddm_Bpert_off[ik]) > 1e-8;
		}
		commutator_mat_diag(double_commut[ik], e_dm[ik], ddm_Bpert_neq[ik], nb_dm, ci); // [ek,[ek,ddm_eq_k]]
		vec3_dot_vec3array(dBS[ik], Bnorm, s[ik], nb_dm*nb_dm);
		DP_precess_fac[ik] = trace_AB(dBS[ik], double_commut[ik], nb_dm); // Tr{sk [ek,[ek,ddm_eq_k]]}
		dstot += trace_AB(dBS[ik], ddm_Bpert[ik], nb_dm); // sum_k Tr(sk ddm_eq_k)
	}

	if (ionode) print_array_atk(has_precess, "has_precess:\n");
	if (ionode) print_mat_atk(ddm_Bpert, nb_dm, "ddm_Bpert:\n");
	if (ionode) print_mat_atk(ddm_Bpert_neq, nb_dm, "ddm_Bpert_neq:\n");
	if (ionode) print_mat_atk(double_commut, nb_dm, "[ek,[ek,ddm_eq_k]]:\n");
	if (ionode) print_array_atk(DP_precess_fac, "Tr{sk [ek,[ek,ddm_eq_k]]}:\n");
	if (ionode) printf("dstot = %lg\n", dstot / nk);

	for (int ik = 0; ik < nk; ik++)
		DP_precess_fac[ik] /= dstot;

	// 1/tau_DP = sum_k {tau^neq_k * Tr{sk [ek,[ek,ddm_eq_k]]}} / sum_k Tr(sk ddm_eq_k)
	if (!imsig_eph_kn) return;
	if (imsig_eph_avg != 0){
		double rate_DP = 0;
		double tau_neq = 0.5 / imsig_eph_avg;
		for (int ik = 0; ik < nk; ik++)
			rate_DP += DP_precess_fac[ik];
		rate_DP *= tau_neq;
		if (ionode) printf("tau_DP with imsig_eph_avg: %lg ps\n", 1 / rate_DP / ps);
	}
	if (imsig_eph_k){
		double rate_DP = 0;
		for (int ik = 0; ik < nk; ik++){
			double tau_neq = 0.5 / imsig_eph_k[ik];
			rate_DP += tau_neq * DP_precess_fac[ik];
		}
		if (ionode) printf("tau_DP with imsig_eph_k: %lg ps\n", 1 / rate_DP / ps);
	}

	for (int ik = 0; ik < nk; ik++)
	for (int ibb = 0; ibb < nb_dm*nb_dm; ibb++)
		ddm_Bpert_neq[ik][ibb] *= 0.5 / imsig_eph_k[ik];
	if (ionode) printf("Tr[ddm_neq ddm_neq] = %lg\n", trace_square_hermite(ddm_Bpert_neq, nk, nb_dm) / nk);
	// dm_neq = f + ddm_eq + ddm_neq
	trunc_copy_arraymat(dm_Bpert_neq, dm_Bpert, nk, nb_dm, 0, nb_dm);
	axbyc(dm_Bpert_neq, ddm_Bpert_neq, nk, nb_dm*nb_dm, c1, c1); // y=ax+by+c, void axbyc(complex **y, complex **x, int n1, int n2, complex a = c1, complex b = c0, complex c = c0);

	if (ionode) printf("**************************************************\n");
}
*/