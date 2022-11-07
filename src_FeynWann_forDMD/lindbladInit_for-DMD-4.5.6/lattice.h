#pragma once
#include "common_headers.h"
#include "parameters.h"

class lattice{
public:
	FeynWann& fw;
	const matrix3<> G, GGT;
	const vector3<> K, Kp;

	lattice(FeynWann& fw, parameters *param):
		fw(fw), G(2 * M_PI * inv(fw.R)), GGT(G * (~G)),
		K(vector3<>(1. / 3, 1. / 3, 0)), Kp(vector3<>(-1. / 3, -1. / 3, 0))
	{}

	bool isKvalley(vector3<> k){
		return GGT.metric_length_squared(wrap_around_Gamma(K - k))
			< GGT.metric_length_squared(wrap_around_Gamma(Kp - k));
	}
	bool isInterValley(vector3<> k1, vector3<> k2){
		return isKvalley(k1) xor isKvalley(k2);
	}

	int dimension(){
		double d = 3;
		for (int iDir = 0; iDir<3; iDir++)
		if (fw.isTruncated[iDir]) d--;
		return d;
	}
	double cell_size(){
		if (dimension() == 3) return fabs(det(fw.R));
		else if (dimension() == 2) return fabs(det(fw.R)) / fw.R(2, 2);
		else if (dimension() == 1) return fw.R(2, 2);
		else return 1;
	}
	double cminvdim2au(){
		return std::pow(bohr2cm, dimension());
	}
	void print_carrier_density(double carrier_density){
		carrier_density /= cminvdim2au();
		if (dimension() == 3)
			logPrintf("Carrier density: %14.7le cm^-3\n", carrier_density);
		else if (dimension() == 2)
			logPrintf("Carrier density: %14.7le cm^-2\n", carrier_density);
		else if (dimension() == 1)
			logPrintf("Carrier density: %14.7le cm^-1\n", carrier_density);
		else
			logPrintf("Carrier density: %14.7le\n", carrier_density);
	}
};