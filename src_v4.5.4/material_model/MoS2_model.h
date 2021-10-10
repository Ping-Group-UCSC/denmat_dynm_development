#pragma once
#include "parameters.h"
#include "MoS2_lattice.h"
#include "MoS2_electron.h"
#include "MoS2_phonon.h"
#include "MoS2_ElectronPhonon.h"

struct mos2_model{
public:
	mos2_model(parameters* param){
		lattice_mos2 *latt = new lattice_mos2(param);
		electron_mos2 *elec = new electron_mos2(param, latt);
		phonon_mos2 *ph = new phonon_mos2(param, latt);
		electronphonon_mos2 *eph = new electronphonon_mos2(latt, param, elec, ph, alg.eph_sepr_eh, !alg.eph_need_elec);
		mpkpair.distribute_var("mos2_model", eph->nkpair_glob);
		eph->set_eph();
	}
};