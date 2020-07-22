#pragma once
#include "GaAs_lattice.h"
#include "electron.h"

class elec_model_GaAs :public electron{
public:
	lattice_GaAs *latt;
	double twomeff, gamma, gs;
	const double ethr;
	vector3<> *kvec;
	complex **evc; // dimension (nk,nb*2). [2*i] for spin up, [2*i+1] for spin down
	double **e0; // for debug
	complex **H_Omega;

	elec_model_GaAs(lattice_GaAs *latt, parameters *param);

	void get_nk();
	inline double e0k_gaas(vector3<>& k);

	void setState0();
	void state0_gaas(vector3<>& k, double e[2], double f[2]);

	void setHOmega_gaas();
	inline vector3<> Omega_gaas(vector3<> k);
	void set_H_BS(int ik0_glob, int ik1_glob);

	void setState_Bso_gaas();

	void smat(complex *v, complex **s);
	// use a perturbation (gs * Bpert \cdot S)
	// currently Bpert has only z component and eigenstates are pure states
	// therefore, dm_Bpert will be fermi-dirac distribution of equilibrium under Bzpert
	void compute_dm_Bpert(vector3<> Bpert);

	void write_ldbd_size();
	void write_ldbd_kvec();
	void write_ldbd_ek();
	void write_ldbd_smat();
	void write_ldbd_dm0();
};