#pragma once
#include "common_headers.h"
#include "lattice.h"
#include "parameters.h"

class phonon{
public:
	lattice *latt;
	double temperature;
	int nq, nm;

	phonon(parameters *param) :temperature(param->temperature){}
	phonon(lattice *latt, parameters *param) :latt(latt), temperature(param->temperature){}

	inline double bose(double t, double w) const
	{
		double wbyt = w / t;
		if (wbyt > 46) return 0;
		if (wbyt < 1e-20) return 0;
		else return 1. / (exp(wbyt) - 1);
	}
};