#pragma once
#include "common_headers.h"
#include "lattice.h"
#include "parameters.h"

class phonon{
public:
	lattice *latt;
	double temperature, omega_max;
	int nq, nm, modeStart, modeEnd;
	qIndexMap *qmap;
	std::vector<vector3<double>> qvec;
	double qmin, qmax;

	phonon(parameters *param) :temperature(param->temperature), modeStart(param->modeStart), modeEnd(param->modeEnd){ init(); }
	phonon(lattice *latt, parameters *param) :latt(latt), temperature(param->temperature), modeStart(param->modeStart), modeEnd(param->modeEnd){ init(); }
	phonon(lattice *latt, parameters *param, electron *elec) :latt(latt), temperature(param->temperature), modeStart(param->modeStart), modeEnd(param->modeEnd){ init(elec); }
	void init(electron *elec = nullptr){
		if (ionode) printf("\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("phonon\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");

		if (material_model == "none") read_jdftx();
		qmin = 100; qmax = 0;
		if (elec != nullptr){
			qmap = new qIndexMap(elec->kmesh); qmap->build(elec->kvec, qvec);
			for (size_t iq = 1; iq < qvec.size(); iq++){ // exclude the first q vector which must be Gamma
				double qlength = latt->klength(qvec[iq]);
				if (qlength < qmin) qmin = qlength;
				if (qlength > qmax) qmax = qlength;
			}
			if (ionode && DEBUG){
				string fnameq = dir_debug + "qIndexMap.out";
				FILE *fpq = fopen(fnameq.c_str(), "w");
				fprintf(fpq, "\nPrint qIndexMap:\n"); fflush(fpq);
				for (size_t iq = 0; iq < qvec.size(); iq++){
					std::map<vector3<int>, size_t>::iterator iter = qmap->the_map.find(qmap->iqvec3(qvec[iq]));
					fprintf(fpq, "iqvec3 = (%d,%d,%d) iq = %lu\n", iter->first[0], iter->first[1], iter->first[2], iter->second);  fflush(fpq);
				}
				fclose(fpq);
			}
		}
		if (ionode) printf("qmin = %14.7le qmax = %14.7le\n", qmin, qmax);
	}

	void read_jdftx(){
		FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
		char s[200];
		for (int i = 0; i < 4; i++)
			fgets(s, sizeof s, fp);
		if (fgets(s, sizeof s, fp) != NULL){
			int itmp1, itmp2, itmp3, itmp4;
			if (sscanf(s, "%d %d %d %d", &itmp1, &itmp2, &itmp3, &itmp4) == 4){
				if (ionode) printf("modeStart = %d modeStop = %d modeSkipStart = %d modeSkipStop = %d from initialization\n", itmp1, itmp2, itmp3, itmp4);
				if (itmp4 - itmp3 > 0) nm = itmp2 - itmp1 - (itmp4 - itmp3);
				else nm = itmp2 - itmp1;
			}
			else if (sscanf(s, "%d %d", &itmp1, &itmp2) == 2){
				if (ionode) printf("modeStart = %d modeStop = %d\n from initialization", itmp1, itmp2);
				nm = itmp2 - itmp1;
			}
			else{
				error_message("sscanf incorrect", "phonon::read_jdftx");
			}
		}
		if (ionode) printf("number of phonon modes = %d\n", nm);
		modeEnd = modeEnd > modeStart ? std::min(nm, modeEnd) : nm;
		modeStart = std::max(0, modeStart);
		if (ionode) printf("modeStart = %d modeEnd = %d (relative to modes from initialization)\n", modeStart, modeEnd);
		for (int i = 0; i < 4; i++)
			fgets(s, sizeof s, fp);
		if (fgets(s, sizeof s, fp) != NULL){
			sscanf(s, "%le", &omega_max); if (ionode) printf("omega_max = %14.7le meV\n", omega_max / eV * 1000);
		}
		fclose(fp);
	}

	inline double bose(double t, double w) const
	{
		double wbyt = w / t;
		if (wbyt > 46) return 0;
		if (wbyt < 1e-20) return 0;
		else return 1. / (exp(wbyt) - 1);
	}
};