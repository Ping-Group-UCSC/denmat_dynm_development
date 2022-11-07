#pragma once
#include "mymp.h"
#include "MoS2_lattice.h"
#include "electron.h"

class electron_mos2 :public electron{
public:
	mymp *mp;
	lattice_mos2 *latt;
	double estart, eend;
	const double twomeff, Omega_z0, A1, A2, alpha_e, gs;
	struct valleyInfo{
		bool isK;
		vector3<> k;
		vector3<> kV;
	} *vinfo;
	double **e0; // for debug
	complex *Uh, *mtmp, **H_Omega;

	electron_mos2(parameters *param, lattice_mos2 *latt);

	void set_erange(double ethr){
		estart = -1e-6; eend = ethr;
	}
	void set_brange(){
		ns = 1; nb = 2; nv = 0; nc = 2;
		nb_dm = nc; nb_eph = nc; nb_wannier = nc;
	}
	void get_nk();
	inline double e0k_mos2(vector3<>& k);

	void setState0();
	void state0_mos2(vector3<>& k, valleyInfo& vinfo, double e[2], double f[2]);

	void setHOmega_mos2();
	inline vector3<> Omega_mos2(valleyInfo& v);

	void setState_Bso_mos2();

	void smat(complex **s);
	void smat(complex *v, complex **s);

	void write_ldbd_size(double degauss);
	void write_ldbd_kvec();
	void write_ldbd_ek();
	void write_ldbd_smat();
	void write_ldbd_U();
	void write_ldbd_Bso();
};