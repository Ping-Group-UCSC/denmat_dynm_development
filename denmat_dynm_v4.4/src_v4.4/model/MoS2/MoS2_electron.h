#pragma once
#include "MoS2_lattice.h"
#include "electron.h"

class elec_model_MoS2 :public electron{
public:
	lattice_MoS2 *latt;
	const double twomeff, Omega_z0, A1, A2, alpha_e, gs;
	const double ethr;
	struct valleyInfo{
		bool isK;
		vector3<> k;
		vector3<> kV;
	} *vinfo;
	complex **evc; // dimension (nk,nb*2). [2*i] for spin up, [2*i+1] for spin down
	double **e0; // for debug
	complex **H_Omega;

	elec_model_MoS2(lattice_MoS2 *latt, parameters *param);

	void get_nk();
	inline double e0k_mos2(vector3<>& k);

	void setState0();
	void state0_mos2(vector3<>& k, valleyInfo& vinfo, double e[2], double f[2]);

	void setHOmega_mos2();
	inline vector3<> Omega_mos2(valleyInfo& v);
	void set_H_BS(int ik0_glob, int ik1_glob);

	void setState_Bso_mos2();

	void smat(complex *v, complex **s);
	// use a perturbation (gs * Bpert \cdot S)
	// currently eigenstates are pure states
	// therefore, dm_Bpert will be fermi-dirac distribution of equilibrium under Bpert
	void compute_dm_Bpert(vector3<> Bpert);

	void write_ldbd_size();
	void write_ldbd_kvec();
	void write_ldbd_ek();
	void write_ldbd_smat();
	void write_ldbd_dm0();
};