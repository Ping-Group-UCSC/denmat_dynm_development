#pragma once
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
//#include <cblas.h>
#include <gsl/gsl_cblas.h>
#include <ODE.h>
#include <mymp.h>
#include <matrix3.h>
#include <Units.h>
#include <myio.h>
#include <constants.h>
#include <myarray.h>
#include <mymatrix.h>
#include <Random.h>
#include <sparse_matrix.h>
#include <sparse2D.h>
#include "Histogram.h"
#include "GaussianSmapling.h"
using namespace std;
using namespace std::chrono;

extern bool ionode;
extern int level_debug_dmdyn;

class algorithm{
public:
	string picture, scatt, ode_method;
	bool H0hasBS, pure, expt, expt_elight, ddmeq, summode, eph_sepr_eh, eph_need_elec, eph_need_hole, eph_enable, phenom_relax, sparseP;
	double thr_sparseP;

	algorithm(){
		H0hasBS = true;
		pure = true; // eigenstates have no spin mixing
		picture = "interaction";
		scatt = "lindblad";
		expt = true;
		expt_elight = true;
		ddmeq = false;
		summode = true;
		eph_sepr_eh = false;
		eph_need_elec = false;
		eph_need_hole = false;
		eph_enable = true;
		phenom_relax = false;
		ode_method = "rkf45";
		sparseP = false;
		thr_sparseP = 1e-40;
	}
};

extern algorithm alg;
extern string code;