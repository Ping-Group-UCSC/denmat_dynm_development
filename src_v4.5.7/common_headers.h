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
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
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
#include "kmap.h"
using namespace std;
using namespace std::chrono;

extern bool ionode;
extern bool DEBUG;
extern string dir_debug;

class algorithm{
public:
	string picture, scatt, ode_method;
	bool expt, expt_elight, ddmdteq, summode, eph_sepr_eh, eph_need_elec, eph_need_hole, sparseP, Pin_is_sparse, set_scv_zero, semiclassical;
	bool modelH0hasBS, read_Bso, scatt_enable, eph_enable, phenom_relax, only_eimp, only_ee, only_intravalley, only_intervalley, linearize, linearize_dPee;
	bool use_dmDP_taufm_as_init, DP_beyond_carrierlifetime, positive_tauneq, use_dmDP_in_evolution;
	double thr_sparseP, mix_tauneq;

	algorithm(){
		picture = "interaction";
		scatt = "lindblad";
		expt = true;
		expt_elight = true;
		ddmdteq = false;
		summode = true;
		linearize = false;
		linearize_dPee = false;
		eph_sepr_eh = false;
		eph_need_elec = false;
		eph_need_hole = false;
		scatt_enable = true;
		eph_enable = true;
		only_eimp = false;
		only_ee = false;
		only_intravalley = false;
		only_intervalley = false;
		phenom_relax = false;
		ode_method = "rkf45";
		Pin_is_sparse = false;
		sparseP = false;
		thr_sparseP = 1e-40;
		set_scv_zero = false;
		semiclassical = false;
		modelH0hasBS = true;
		use_dmDP_taufm_as_init = false;
		DP_beyond_carrierlifetime = false;
		positive_tauneq = false;
		use_dmDP_in_evolution = false;
		mix_tauneq = 0.2;
		read_Bso = false;
	}
};

extern algorithm alg;
extern string code;
extern string material_model;