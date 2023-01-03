#pragma once
#include "common_headers.h"
#include "parameters.h"

class lattice{
public:
	int dim;
	double a, volume, area, thickness, length, density, cell_size, cell_size_cmd, cmd;
	matrix3<> R, Gvec, GGT;
	vector3<> K, Kp;
	std::vector<vector3<>> vpos;
	std::vector<std::vector<bool>> vtrans;
	string type_q_ana;

	lattice(parameters *param)
		: R(param->R), vpos(param->vpos), vtrans(param->vtrans), type_q_ana(param->type_q_ana)
	{
		if (ionode) printf("\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("lattice\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");

		if (code == "jdftx") read_ldbd_R();

		volume = fabs(det(R));
		area = volume / R(2, 2);
		thickness = param->thickness > 1e-6 ? param->thickness : R(2, 2);
		length = R(2, 2);

		cell_size = volume;
		if (dim == 2) cell_size = area;
		if (dim == 1) cell_size = length;
		if (dim == 0) cell_size = 1;
		double cmd = std::pow(bohr2cm, dim);
		cell_size_cmd = cell_size * cmd;
		if (ionode) printf("cell_size = %lg a.u.\n", cell_size);
		if (cell_size == 0) error_message("cell_size is zero", "lattice");

		Gvec = (2.*M_PI)*inv(R);
		GGT = Gvec * (~Gvec);

		K[0] = 1. / 3; K[1] = 1. / 3; K[2] = 0;
		Kp[0] = -1. / 3; Kp[1] = -1. / 3; Kp[2] = 0;
	}

	void read_ldbd_R(){
		if (ionode) printf("read ldbd_R.dat:\n");
		if (FILE *fp = fopen("ldbd_data/ldbd_R.dat", "r")) {
			fscanf(fp, "%d", &dim); if (ionode) printf("dim = %d\n", dim);
			for (int i = 0; i < 3; i++)
				fscanf(fp, "%lf %lf %lf", &R(i, 0), &R(i, 1), &R(i, 2));
			fclose(fp);
		}
	}
	
	void printLattice(){
		string s;
		s = "R"; printMatrix3(R, s);
		s = "Gvec"; printMatrix3(Gvec, s);
	}

	void printMatrix3(matrix3 <>& m, string& s){
		if (ionode){
			printf("%s\n", s.c_str());
			printf("%lg %lg %lg\n", m(0, 0), m(0, 1), m(0, 2));
			printf("%lg %lg %lg\n", m(1, 0), m(1, 1), m(1, 2));
			printf("%lg %lg %lg\n", m(2, 0), m(2, 1), m(2, 2));
		}
	}

	inline double klengthSq(vector3<> k, vector3<> center = vector3<>(0,0,0)) const
	{
		return GGT.metric_length_squared(wrap(k, center));
	}
	inline double klength(vector3<> k, vector3<> center = vector3<>(0, 0, 0)) const
	{
		return sqrt(GGT.metric_length_squared(wrap(k, center)));
	}

	inline int whichvalley(vector3<> k) const
	{
		if (vpos.size() < 2) return -1;
		int iv = 0;
		double klSqmin = klengthSq(k - vpos[0]);
		for (int i = 1; i < vpos.size(); i++){
			double klSq = klengthSq(k - vpos[i]);
			if (klSq < klSqmin){
				iv = i;
				klSqmin = klSq;
			}
		}
		return iv;
	}

	inline bool isKvalley(vector3<> k) const
	{
		return GGT.metric_length_squared(wrap(k - K))
			< GGT.metric_length_squared(wrap(k - Kp));
	}
	bool isIntravalley(vector3<> k, vector3<> kp) const
	{
		bool kisK = isKvalley(k), kpisK = isKvalley(kp);
		return (kisK && kpisK) || (!kisK && !kpisK);
	}
	// latter two functions for model Hamiltonians
	inline bool isKvalley(vector3<> k, vector3<>& kV, double& kVSq) const
	{
		vector3<> kK = wrap(k - K);
		double kKSq = GGT.metric_length_squared(kK);
		vector3<> kKp = wrap(k - Kp);
		double kKpSq = GGT.metric_length_squared(kKp);
		bool isK = kKSq < kKpSq;
		if (isK){
			kV = kK;
			kVSq = kKSq;
		}
		else{
			kV = kKp;
			kVSq = kKpSq;
		}
		return isK;
	}
	inline double ktoKorKpSq(vector3<> k) const
	{
		double ktoKSq = GGT.metric_length_squared(wrap(k - K));
		double ktoKpSq = GGT.metric_length_squared(wrap(k - Kp));
		return std::min(ktoKSq, ktoKpSq);
	}

	bool kpair_is_allowed(vector3<> k1, vector3<> k2, int& iv1, int& iv2){
		bool isIntravellay = isIntravalley(k1, k2);
		if (isIntravellay && alg.only_intervalley) return false;
		if (!isIntravellay && alg.only_intravalley) return false;

		iv1 = whichvalley(k1);
		iv2 = whichvalley(k2);
		if (iv1 >= 0 && iv2 >= 0 && !vtrans[iv1][iv2]) return false;
		return true;
	}

	double qana(vector3<> k1, vector3<> k2, int iv1 = 0, int iv2 = 0){
		double qlength = 0;
		if (alg.only_intervalley){
			if (type_q_ana != "wrap_around_minusk")
				qlength = sqrt(ktoKorKpSq(k1 - k2));
			else
				qlength = klength(wrap(k1) + wrap(k2));
		}
		else if (iv1 >= 0 && iv2 >= 0){
			vector3<> dv = wrap(vpos[iv1] - vpos[iv2]);
			if (type_q_ana != "wrap_around_minusk"){
				vector3<> q = wrap(k1 - k2);
				double q_to_dv = klength(q, dv);
				double q_to_mdv = klength(q, -dv);
				qlength = std::min(q_to_dv, q_to_mdv);
			}
			else{
				if (klength(dv) > 1e-10)
					error_message("two valleys are not opposite to each other for type_q_ana = wrap_around_minusk", "lattic::qana");
				qlength = klength(wrap(k1) + wrap(k2));
			}
		}
		else{
			qlength = klength(k1 - k2);
		}
		return qlength;
	}
};