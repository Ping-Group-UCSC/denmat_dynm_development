#pragma once
#include "common_headers.h"
#include "Scatt_Param.h"
#include "ElecImp_Model.h"

class electronimpurity{
public:
	int iD; //which defect
	elecimp_model *eimp_model;
	mymp *mp;
	bool isHole;
	FILE *fp1, *fp2;
	sparse2D *sP1, *sP2;
	int nkpair_glob, nb, nbpow4;
	double fraction;

	electronimpurity(int iD, mymp *mp, bool isHole, int nkpair_glob, int nb, double volume)
		: iD(iD), eimp_model(nullptr), mp(mp), isHole(isHole), nkpair_glob(nkpair_glob), nb(nb), nbpow4((int)std::pow(nb, 4)), fraction(abs(eip.ni[iD] * volume)),
		sP1(nullptr), sP2(nullptr)
	{
		if (ionode) printf("init ab initio elec-imp\n");
		if (ionode) printf("fraction = %lg\n", fraction);
	}

	void read_ldbd_imp_P(int ik, complex *P1, complex *P2); // ik means ikpair
};