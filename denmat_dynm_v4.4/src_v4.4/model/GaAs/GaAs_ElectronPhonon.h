#pragma once
#include "GaAs_lattice.h"
#include "GaAs_electron.h"
#include "GaAs_phonon.h"
#include "ElectronPhonon.h"
#include "mymp.h"

class eph_model_GaAs :public electronphonon{
public:
	lattice_GaAs *latt;
	elec_model_GaAs *elec_gaas;
	ph_model_GaAs *ph_gaas;
	const double eps_inf, eps0, alpha, xi, e14;
	double prefac_ta, prefac_la, prefac_lo;

	eph_model_GaAs(lattice_GaAs *latt, parameters *param, elec_model_GaAs *elec_gaas, ph_model_GaAs *ph_gaas, bool sepr_eh = false, bool isHole = false)
		: latt(latt), electronphonon(param), elec_gaas(elec_gaas), ph_gaas(ph_gaas),
		eps_inf(10.8), eps0(12.9), alpha(1. / sqrt(2.*ph_gaas->wlo) * (1. / eps_inf - 1. / eps0)), xi(7 * eV), e14(1.41e9*Volt / meter)
	{
		nk_glob = elec_gaas->nk;
		nb = elec_gaas->nb;
		bStart = 0; bEnd = nb; nb_expand = nb;
		nm = ph_gaas->nm;
		prefac_gauss = 1. / sqrt(2 * M_PI) * param->degauss;
		prefac_sqrtgauss = 1. / sqrt(sqrt(2 * M_PI) * param->degauss);
		prefac_eph = 2 * M_PI / elec_gaas->nk_full;
		prefac_ta = 4 * sqrt(2) *M_PI*e14 / eps0 / sqrt(latt->density*ph_gaas->vst);
		prefac_la = sqrt(xi*xi / 2 / latt->density / ph_gaas->vsl);
		prefac_lo = sqrt(4 * M_PI*alpha*std::pow(ph_gaas->wlo, 1.5) / sqrt(2));
		
		alloc_nonparallel();
		this->e = elec_gaas->e;

		get_nkpair();
	}

	void set_eph(mymp *mp);
	void get_nkpair(){
		nkpair_glob = nk_glob * nk_glob;
	}
	void set_kpair();
	void write_ldbd_kpair();
	void set_ephmat();
	void write_ldbd_eph();
	void merge_eph();
	void debug_eph_model_GaAs();
	inline void g_model_GaAs(vector3<> q, int im, complex vk[2*2], complex vkp[2*2], complex g[2 * 2]);

	void evolve_debug(double t, complex **dm, complex **dm1, complex **ddm);
};