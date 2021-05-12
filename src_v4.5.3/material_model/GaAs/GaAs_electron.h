#pragma once
#include "mymp.h"
#include "GaAs_lattice.h"
#include "electron.h"

class electron_gaas :public electron{
public:
	mymp *mp;
	lattice_gaas *latt;
	double estart, eend;
	const double twomeff, twogammaD;
	double **e0; // for debug
	complex *Uh, *mtmp, **H_Omega;

	electron_gaas(parameters *param, lattice_gaas *latt);

	void set_erange(double ethr){
		estart = -1e-6; eend = ethr;
	}
	void set_brange(){
		ns = 1; nb = 2; nv = 0; nc = 2;
		nb_dm = nc; nb_eph = nc; nb_wannier = nc;
	}
	void get_nk();
	inline double e0k_gaas(vector3<>& k);

	void setState0();

	void setHOmega_gaas();
	inline vector3<> Omega_gaas(vector3<>& k);

	void setState_Bso_gaas();

	void smat(complex **s);
	void smat(complex *v, complex **s);

	void write_ldbd_size(double degauss);
	void write_ldbd_kvec();
	void write_ldbd_ek();
	void write_ldbd_smat();
	void write_ldbd_U();
	void write_ldbd_Bso();
};