#pragma once
#include "GaAs_lattice.h"
#include "GaAs_electron.h"
#include "GaAs_phonon.h"
#include "ElectronPhonon.h"

class electronphonon_gaas :public electronphonon{
public:
	mymp *mp;
	lattice_gaas *latt;
	electron_gaas *elec_gaas;
	phonon_gaas *ph_gaas;
	std::vector<std::pair<size_t, size_t>> kpairs;

	electronphonon_gaas(lattice_gaas *latt, parameters *param, electron_gaas *elec_gaas, phonon_gaas *ph_gaas, bool sepr_eh = false, bool isHole = false)
		:mp(&mpkpair), latt(latt), electronphonon(param), elec_gaas(elec_gaas), ph_gaas(ph_gaas)
	{
		nk_glob = elec_gaas->nk;
		nb = elec_gaas->nb;
		bStart = 0; bEnd = nb; nb_expand = nb;
		nm = ph_gaas->nm;
		prefac_eph = 2 * M_PI / elec_gaas->nk_full;
		
		alloc_nonparallel();
		this->e = elec_gaas->e;

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
	inline void g_model_gaas(double q, int im, double wq, complex vk[2*2], complex vkp[2*2], complex g[2 * 2]);
};