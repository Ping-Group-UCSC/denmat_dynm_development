#pragma once
#include "common_headers.h"
#include "lattice.h"
#include "parameters.h"
#include "PumpProbe.h"
#include "Scatt_Param.h"

class electron{
public:
	mymp *mp, *mp_morek;
	lattice *latt;
	double temperature, mu, carrier_density, ne, nh; bool carrier_density_means_excess_density;
	vector3<int> kmesh; double nk_full;
	int ns, nk, nb, nb_wannier, bskipped_wannier, bskipped_dft, nk_morek; // spin, k point, band
	int nv, nc; // valence and conduction bands
	int bStart_dm, bEnd_dm, nb_dm, nv_dm, nc_dm; // band range related to density matrix
	int bStart_eph, bEnd_eph, nb_eph, nv_eph, nc_eph;
	std::vector<vector3<>> kvec, kvec_morek;
	bool print_along_kpath, print_layer_occ, print_layer_spin;
	std::vector<vector3<double>> kpath_start, kpath_end;
	int nkpath;
	std::vector<int> ik_kpath;
	double **e, **f, **e_dm, **f_dm, **e_morek, **f_morek, **e_dm_morek, **f_dm_morek; // ocuupation number
	double emin, emax, evmax, ecmin, emid, scissor;
	vector3<> B; double scale_Ez;
	bool gfac_normal_dist, gfac_k_resolved;
	double gfac_mean, gfac_sigma, gfac_cap;
	double *gfack; // mu_B * g of states
	bool needL;
	complex ***s, **layer, **layerspin, ***v, **U, ***l;
	std::vector<vector3<>> Bso;
	complex **H_BS, **H_Ez;
	double degthr;
	complex **ddm_Bpert, **dm_Bpert, **ddm_Bpert_neq, **dm_Bpert_neq; // store ddm_eq and ddm_neq for later analysis of ddm
	bool *has_precess;
	double trace_sq_ddm_tot;
	//double *DP_precess_fac;
	double **imsig_eph_kn, *imsig_eph_k, imsig_eph_avg;
	double ***b2kn, ***b2ew; double **b2w_avg; // b2ew, b2w_avg: frequency-dependent "scattering" spin-mixing

	bool rotate_spin_axes;
	vector3<double> sdir_z;
	matrix3<double> sdir_rot; // redefined spin directions

	electron(parameters *param)
		:temperature(param->temperature), mu(param->mu), carrier_density(param->carrier_density), carrier_density_means_excess_density(param->carrier_density_means_excess_density),
		kmesh(vector3<int>(param->nk1, param->nk2, param->nk3)), nk_full((double)param->nk1*(double)param->nk2*(double)param->nk3), B(param->B),
		print_along_kpath(param->print_along_kpath), kpath_start(param->kpath_start), kpath_end(param->kpath_end), nkpath(param->kpath_start.size()),
		needL(param->needL), scissor(param->scissor),
		rotate_spin_axes(param->rotate_spin_axes), sdir_z(param->sdir_z), sdir_rot(param->sdir_rot),
		v(nullptr){}
	electron(mymp *mp, mymp *mp_morek, lattice *latt, parameters *param)
		:mp(mp), mp_morek(mp_morek), latt(latt), temperature(param->temperature), mu(param->mu), carrier_density(param->carrier_density), carrier_density_means_excess_density(param->carrier_density_means_excess_density),
		kmesh(vector3<int>(param->nk1, param->nk2, param->nk3)), nk_full((double)param->nk1*(double)param->nk2*(double)param->nk3), B(param->B), scale_Ez(param->scale_Ez),
		print_along_kpath(param->print_along_kpath), kpath_start(param->kpath_start), kpath_end(param->kpath_end), nkpath(param->kpath_start.size()),
		H_BS(nullptr), H_Ez(nullptr), ddm_Bpert(nullptr), ddm_Bpert_neq(nullptr), dm_Bpert_neq(nullptr),
		imsig_eph_kn(nullptr), imsig_eph_k(nullptr), imsig_eph_avg(0),
		gfack(nullptr), gfac_normal_dist(param->gfac_normal_dist), gfac_k_resolved(param->gfac_k_resolved), gfac_mean(param->gfac_mean), gfac_sigma(param->gfac_sigma), gfac_cap(param->gfac_cap),
		needL(param->needL), scissor(param->scissor),
		rotate_spin_axes(param->rotate_spin_axes), sdir_z(param->sdir_z), sdir_rot(param->sdir_rot),
		v(nullptr)
	{
		if (ionode) printf("\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("electron\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");

		if (code == "jdftx"){
			read_ldbd_size();
			bool alloc_v = exists("ldbd_data/ldbd_vmat.bin"); //false;
			if (pmp.laserA > 0 and !alloc_v) error_message("laserA > 0 and !alloc_v", "electron constructor");
			bool alloc_U = exists("ldbd_data/ldbd_Umat.bin"); //false;
			print_layer_occ = exists("ldbd_data/ldbd_layermat.bin");
			print_layer_spin = exists("ldbd_data/ldbd_layerspinmat.bin");
			//for (int iD = 0; iD < eip.ni.size(); iD++)
			//	if (eep.eeMode != "none" || (eip.ni[iD] != 0 && eip.impMode[iD] == "model_ionized")) alloc_U = true;
			alloc_mat(alloc_v, alloc_U);
			read_ldbd_kvec();
			if (print_along_kpath && kvec.size() > 0) get_kpath();
			read_ldbd_ek();
			e_dm = trunc_alloccopy_array(e, nk, bStart_dm, bEnd_dm);
			f_dm = trunc_alloccopy_array(f, nk, bStart_dm, bEnd_dm);
			//if (alg.eph_enable) read_ldbd_imsig_eph();
			read_ldbd_smat();
			if (needL) read_ldbd_lmat();
			if (print_layer_occ) read_ldbd_layermat();
			if (print_layer_occ) read_ldbd_layerspinmat();
			if (pmp.laserA > 0) read_ldbd_vmat();
			if (alloc_U) read_ldbd_Umat();
			if (alg.read_Bso) read_ldbd_Bso();
		}
		emax = maxval(e_dm, nk, nv_dm, nb_dm);
		ecmin = minval(e_dm, nk, nv_dm, nb_dm);
		if (ionode) printf("\nElectronic states energy range:\n");
		if (nb_dm > nv_dm && ionode) printf("emax = %lg ecmin = %lg\n", emax, ecmin);
		for (int iD = 0; iD < eip.ni.size(); iD++)
			if (nb_dm > nv_dm && eip.partial_ionized[iD] && eip.Eimp[iD] > ecmin) error_message("Impurity level should not be higher than CBM", "electron constructor");
		evmax = maxval(e_dm, nk, 0, nv_dm);
		emin = minval(e_dm, nk, 0, nv_dm);
		emid = (ecmin + evmax) / 2;
		if (nv_dm > 0 && ionode) printf("evmax = %lg emin = %lg\n", evmax, emin);
		if (nb_dm == nv_dm) emax = evmax + 1;
		if (nv_dm == 0) emin = ecmin - 1;
		for (int iD = 0; iD < eip.ni.size(); iD++){
			if (nv_dm > 0 && eip.partial_ionized[iD] && eip.Eimp[iD] < evmax) error_message("Impurity level should not be lower than VBM", "electron constructor");
			eip.ni_bvk[iD] = eip.ni[iD] * nk_full * latt->cell_size;
			if (nv_dm > 0 && nb_dm > nv_dm && eip.partial_ionized[iD]){
				if (eip.ni[iD] > 0 && eip.Eimp[iD] < emid) error_message("impurity level < middle energy in gap in n-type", "electron constructor");
				if (eip.ni[iD] < 0 && eip.Eimp[iD] > emid) error_message("impurity level > middle energy in gap in p-type", "electron constructor");
			}
		}
	}
	void alloc_mat(bool alloc_v, bool alloc_U);

	void get_kpath();
	vector3<> get_kvec(int&, int&, int&);
	static double fermi(double t, double mu, double e){
		double ebyt = (e - mu) / t;
		if (ebyt < -46) return 1;
		else if (ebyt > 46) return 0;
		else return 1. / (exp(ebyt) + 1);
	}
	void compute_f(double t, double mu, int nk, mymp *mp, double **e, double **f, double **f_dm){
		zeros(f, nk, nb);
		for (int ik = mp->varstart; ik < mp->varend; ik++)
		for (int i = 0; i < nb; i++)
			f[ik][i] = fermi(t, mu, e[ik][i]);
		mp->allreduce(f, nk, nb, MPI_SUM);
		trunc_copy_array(f_dm, f, nk, bStart_dm, bEnd_dm);
	}
	void compute_f(double t, double mu){
		if (nk_morek > 0) compute_f(t, mu, nk_morek, mp_morek, e_morek, f_morek, f_dm_morek);
		compute_f(t, mu, nk, mp, e, f, f_dm);
	}
	double find_mu(double carrier_bvk, bool is_excess, double t, double mu0);
	double find_mu(double carrier_bvk, bool is_excess, double t, double emin, double emax);
	double compute_nfree(bool isHole, double t, double mu, mymp *mp, double **e);
	double compute_nfree(bool isHole, double t, double mu);
	double set_mu_and_n(double carrier_density){
		//if (carrier_density != 0 && !carrier_density_means_excess_density) mu = find_mu(carrier_density * nk_full * latt->cell_size, false, temperature, mu);
		//if (carrier_density_means_excess_density) mu = find_mu(carrier_density * nk_full * latt->cell_size, true, temperature, mu);
		if (carrier_density != 0 && !carrier_density_means_excess_density) mu = find_mu(carrier_density * nk_full * latt->cell_size, false, temperature, emin, emax);
		if (carrier_density_means_excess_density) mu = find_mu(carrier_density * nk_full * latt->cell_size, true, temperature, emin, emax);
		compute_f(temperature, mu);
		if (ionode) print_array_atk(f, nb, "f:");
		if (ionode) print_array_atk(f_dm, nb_dm, "f_dm:");
		ne = compute_nfree(false, temperature, mu) / nk_full / latt->cell_size;
		nh = compute_nfree(true, temperature, mu) / nk_full / latt->cell_size;
		double nei = eip.compute_carrier_bvk_of_impurity_level(false, temperature, mu) / nk_full / latt->cell_size;
		double nhi = eip.compute_carrier_bvk_of_impurity_level(true, temperature, mu) / nk_full / latt->cell_size;
		if (ionode) printf("ne = %lg nei = %lg ne+nei = %lg\n", ne, nei, ne + nei);
		if (ionode) printf("nh = %lg nhi = %lg nh+nhi = %lg\n", nh, nhi, nh + nhi);
		eip.carrier_bvk_gs = fabs(ne + nei) > fabs(nh + nhi) ? ne + nei : nh + nhi;
		eip.carrier_bvk_gs = eip.carrier_bvk_gs * nk_full * latt->cell_size;
		eip.ne_bvk_gs = ne * nk_full * latt->cell_size; eip.nh_bvk_gs = nh * nk_full * latt->cell_size; clp.nfreetot = fabs(ne) + fabs(nh);
		eip.calc_ni_ionized(temperature, mu);
		return mu;
	}

	void read_ldbd_size();
	void read_ldbd_kvec();
	void read_ldbd_ek();
	void read_ldbd_imsig_eph();
	void read_ldbd_smat();
	void read_ldbd_lmat();
	void read_ldbd_layermat();
	void read_ldbd_layerspinmat();
	void read_ldbd_vmat();
	void read_ldbd_Umat();
	void read_ldbd_Bso();
	void print_array_atk(bool *a, string s = "");
	void print_array_atk(double *a, string s = "", double unit = 1);
	void print_array_atk(double **a, int n, string s = "", double unit = 1);
	void print_mat_atk(complex **m, int n, string s = "");

	void compute_dm_Bpert_1st(vector3<> Bpert, double t0);
	//void compute_DP_related(vector3<> Bpert);
	void deg_proj(complex *m, double *e, int n, double thr, complex *mdeg);
	void compute_b2(double de, double degauss, double degthr);
	void compute_b2(double de, double degauss, double degthr, bool rotate_spin_axes);
	void compute_Bin2();
	void compute_Bin2(bool rotate_spin_axes);
	double average_dfde(double **arr, double **f, int n1, int nb, bool inv = false);

	void set_gfac();
	void set_H_BS(int ik0_glob, int ik1_glob);
	void set_H_Ez(int ik0_glob, int ik1_glob);
};