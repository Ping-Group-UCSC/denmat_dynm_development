#pragma once
#include "lattice.h"

class lattice_mos2 :public lattice{
public:
	lattice_mos2(parameters *param) :lattice(param) {
		dim = 2;
		a = 5.934;
		R(0, 0) = a; R(0, 1) = -0.5 * a; R(1, 1) = 0.5*sqrt(3)*a; R(2, 2) = 20;

		cell_size = area = 30.5;
		thickness = 11.6;
		density = 9530;

		Gvec = (2.*M_PI)*inv(R);
		GGT = Gvec * (~Gvec);

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
