#pragma once
#include "GaAs_lattice.h"
#include "phonon.h"

class phonon_gaas :public phonon{
public:
	lattice_gaas *latt;

	phonon_gaas(parameters *param, lattice_gaas *latt)
		:latt(latt), phonon(param)
	{
		nm = 0;
	}

	inline double omega_model_gaas(double q, int m) const
	{
		return 0;
	}
};