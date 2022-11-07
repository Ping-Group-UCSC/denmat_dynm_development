#pragma once
#include "parameters.h"
#include "GaAs_lattice.h"
#include "GaAs_electron.h"
#include "GaAs_phonon.h"
#include "GaAs_ElectronPhonon.h"

struct gaas_model{
public:
	gaas_model(parameters* param){
		lattice_gaas *latt = new lattice_gaas(param);
		electron_gaas *elec = new electron_gaas(param, latt);
		phonon_gaas *ph = new phonon_gaas(param, latt);
		electronphonon_gaas *eph = new electronphonon_gaas(latt, param, elec, ph, alg.eph_sepr_eh, !alg.eph_need_elec);
		mpkpair.distribute_var("gaas_model", eph->nkpair_glob);
		eph->set_eph();
	}
};