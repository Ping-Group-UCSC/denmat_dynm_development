#pragma once
#include "lattice.h"

class lattice_GaAs :public lattice{
public:
	lattice_GaAs(parameters *param) :lattice(param) {
		dim = 3;
		density = 5300 * kg/std::pow(meter, 3);

		a = 10.6949092647639637;
		R(0, 0) = 0.0 * a; R(0, 1) = 0.5 * a; R(0, 2) = 0.5 * a;
		R(1, 0) = 0.5 * a; R(1, 1) = 0.0 * a; R(1, 2) = 0.5 * a;
		R(2, 0) = 0.5 * a; R(2, 1) = 0.5 * a; R(2, 2) = 0.0 * a;
		Gvec = (2.*M_PI)*inv(R);
		GGT = Gvec * (~Gvec);
		volume = fabs(det(R));
		double m_uc = (69.723 + 74.9216)*1822.88848620953;
		if (ionode) printf("density_set = %lg density_calc = %lg", density, m_uc / volume);

		write_lbdb_R();
	}

	void write_lbdb_R(){
		if (ionode){
			system("mkdir ldbd_data");
			FILE *fp = fopen("ldbd_data/ldbd_R.dat", "w");
			fprintf(fp, "%d\n", dim);
			fprintf(fp, "%14.7le %14.7le %14.7le\n", R(0, 0), R(0, 1), R(0, 2));
			fprintf(fp, "%14.7le %14.7le %14.7le\n", R(1, 0), R(1, 1), R(1, 2));
			fprintf(fp, "%14.7le %14.7le %14.7le\n", R(2, 0), R(2, 1), R(2, 2));
			fclose(fp);
		}
	}
};