#pragma once
#include "MoS2_lattice.h"
#include "MoS2_electron.h"
#include "MoS2_phonon.h"
#include "ElectronPhonon.h"

class electronphonon_mos2 :public electronphonon{
public:
	mymp *mp;
	lattice_mos2 *latt;
	electron_mos2 *elec_mos2;
	phonon_mos2 *ph_mos2;
	std::vector<std::pair<size_t, size_t>> kpairs;
	const double prefac_g, xita,xila, d1tainter, d1lainter, d1tointra, d1tointer, d0lointer, gfr, halfdfr;

	electronphonon_mos2(lattice_mos2 *latt, parameters *param, electron_mos2 *elec_mos2, phonon_mos2 *ph_mos2, bool sepr_eh = false, bool isHole = false)
		:mp(&mpkpair), latt(latt), electronphonon(param), elec_mos2(elec_mos2), ph_mos2(ph_mos2),
		prefac_g(1. / sqrt(2.*(latt->area)*(latt->density))),
		xita(0.0588), xila(0.103), d1tainter(0.217), d1lainter(0.143), d1tointra(0.147), d1tointer(0.0698),
		d0lointer(0.0596), gfr(0.0036), halfdfr(8.33 / 2)
	{
		nk_glob = elec_mos2->nk;
		nb = elec_mos2->nb;
		bStart = 0; bEnd = nb; nb_expand = nb;
		nm = ph_mos2->nm;
		prefac_eph = 2 * M_PI / elec_mos2->nk_full;
		
		alloc_nonparallel();
		this->e = elec_mos2->e;

		get_nkpair();
	}

	void set_eph();
	void get_nkpair(){
		nkpair_glob = nk_glob * (nk_glob + 1) / 2;
	}
	void set_kpair();
	void write_ldbd_kpair();
	void set_ephmat();
	void write_ldbd_eph();
	void write_ldbd_eph(string which);
	inline void g_model_mos2(bool intra, double q, double qV, int im, double wq, complex vk[2*2], complex vkp[2*2], complex g[2 * 2]);
};