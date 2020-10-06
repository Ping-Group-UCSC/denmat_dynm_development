#pragma once
#include "MoS2_lattice.h"
#include "phonon.h"

class ph_model_MoS2 :public phonon{
public:
	lattice_MoS2 *latt;
	const double cta, cla, wto, wlo;

	ph_model_MoS2(lattice_MoS2 *latt, parameters *param)
		:latt(latt), phonon(param),
		cta(0.00192), cla(0.00306), wto(0.00176), wlo(0.00176)
	{
		nm = 4;
	}

	inline double omega_model_MoS2(double q, int m) const
	{
		if (m == 0){
			return cta * q;
		}
		else if (m == 1){
			return cla * q;
		}
		else if (m == 2 || m == 3){
			return wto;
		}
		else{
			error_message("this mode is not allowed");
		}
	}
};