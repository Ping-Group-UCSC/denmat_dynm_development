#pragma once
#include "GaAs_lattice.h"
#include "phonon.h"

class ph_model_GaAs :public phonon{
public:
	lattice_GaAs *latt;
	const double vst, vsl, wlo;

	ph_model_GaAs(lattice_GaAs *latt, parameters *param)
		:latt(latt), phonon(param),
		vst(2480 * meter / sec), vsl(5290 * meter / sec), wlo(0.0354 * eV)
	{
		nm = 3;
	}

	inline double omega_model_GaAs(double q, int m) const
	{
		if (m == 0){
			return vst * q;
		}
		else if (m == 1){
			return vsl * q;
		}
		else if (m == 2){
			return wlo;
		}
		else{
			error_message("this mode is not allowed");
		}
	}
};