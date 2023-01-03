#pragma once
#include "common_headers.h"

struct pumpprobeParameters{
public:
	string laserAlg, laserMode;
	double laserA, laserE, pumpTau, pump_tcenter;
	string laserPoltype;
	vector3<complex> laserPol;

	int probeNE;
	double probeEmin, probeEmax, probeDE, probeTau;
	std::vector<string> probePoltype;
	std::vector<vector3<complex>> probePol;

	bool active(){
		return fabs(laserA) > 1e-10;
	}

	vector3<complex> set_Pol(string s){
		if (s == "LC")
			return normalize(complex(1, 0)*vector3<>(0, 1, 0) + complex(0, 1)*vector3<>(1, 0, 0));
		else if (s == "RC")
			return normalize(complex(1, 0)*vector3<>(1, 0, 0) + complex(0, 1)*vector3<>(0, 1, 0));
		else if (s == "Ex")
			return normalize(complex(1, 0)*vector3<>(1, 0, 0));
		else if (s == "Ey")
			return normalize(complex(1, 0)*vector3<>(0, 1, 0));
		else
			error_message("laserPoltype must be LC, RC, Ex or Ey");
	}

	inline void print(const vector3<complex>& v, const char* format = "%lg "){
		printf("[ "); for (int k = 0; k<3; k++) printf(format, v[k].real()); printf("] + 1j*");
		printf("[ "); for (int k = 0; k<3; k++) printf(format, v[k].imag()); printf("]\n");
	}
	inline vector3<complex> normalize(const vector3<complex>& v) { return v * (1. / sqrt(v[0].norm() + v[1].norm() + v[2].norm())); }
};

extern pumpprobeParameters pmp;