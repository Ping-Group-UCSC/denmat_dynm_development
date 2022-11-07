#pragma once
#include "lattice.h"

class lattice_gaas :public lattice{
public:
	lattice_gaas(parameters *param) :lattice(param) {
		dim = 3;
		a = 10.6829;
		R(0, 0) = 0; R(0, 1) = 0.5 * a; R(0, 2) = 0.5 * a;
		R(1, 0) = 0.5 * a; R(1, 1) = 0; R(1, 2) = 0.5 * a;
		R(2, 0) = 0.5 * a; R(2, 1) = 0.5 * a; R(2, 2) = 0;

		cell_size = volume = fabs(det(R));
		if (ionode) printf("volume = %lg\n", volume);
		if (volume == 0) error_message("volume is zero", "lattice");
		area = volume / R(2, 2);
		thickness = param->thickness > 1e-6 ? param->thickness : R(2, 2);

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
