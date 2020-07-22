#pragma once
#include "common_headers.h"
#include "parameters.h"
#include "lattice.h"
#include "electron.h"
#include "phonon.h"
#include "PumpProbe.h"
#include "ElecLight.h"
#include "ElectronPhonon.h"
#include "phenomenon_relax.h"
#include "DenMat.h"
#include "observable.h"
#include "model.h"

template<class Tl, class Te, class Telight, class Teph>
int func(double t, const double y[], double dydt[], void *params);

template<class Tl, class Te, class Telight, class Teph>
class dm_dynamics{
public:
	Tl* latt;
	parameters* param;
	Te* elec;
	Telight* elight;
	Teph* eph;
	phenom_relax* phnm_rlx;
	singdenmat_k* sdmk;
	ob_1dmk<Tl, Te>* ob;
	int it0;

	dm_dynamics(Tl* latt, parameters* param, Te* elec, Telight* elight, Teph* eph)
		: latt(latt), param(param), elec(elec), elight(elight), eph(eph), it0(param->restart ? 0 : 1)
	{
		// density matrix
		sdmk = new singdenmat_k(param, &mpk, elec->nk, elec->nb_dm); // k-independent single density matrix
		if (!alg.H0hasBS) sdmk->init_Hcoh(elec->H_BS, elec->e_dm);
		// probe ground state
		sdmk->init_dm(elec->f_dm);
		if (pmp.active()) elight->probe(-1, sdmk->t, sdmk->dm, sdmk->oneminusdm);
		// initialize density matrix with spin inblance
		if (param->restart)
			sdmk->read_dm_restart();
		else{
			if (pmp.active() && elight->pumpMode == "perturb"){
				if (ionode) printf("initial density matrix inbalance is induced by pump (Gaussian) pulse perturbation\n");
				sdmk->init_dm(elight->dm_pump);
			}
			else{
				if (ionode) printf("no initial density matrix inbalance or that induced by Bpert (1st order)\n");
				sdmk->init_dm(elec->dm_Bpert);
			}
		}
		if (!(pmp.active() && (elight->pumpMode == "lindblad" || elight->pumpMode == "coherent"))){
			if (alg.ddmeq || alg.phenom_relax) sdmk->set_dm_eq(param->temperature, elec->e_dm, elec->nv_dm);
			if (alg.ddmeq) eph->compute_ddm_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
		}
		//sdmk->write_dm(); sdmk->write_dm_tofile(sdmk->t);

		// phenomenon 
		if (alg.phenom_relax) phnm_rlx = new phenom_relax(param, elec->nk, elec->nb_dm, sdmk->dm_eq);

		// observables
		ob = new ob_1dmk<Tl, Te>(latt, param, elec);
		if (!param->restart){
			ob->measure("dos", false, true, true, sdmk->t, sdmk->dm);
			ob->measure("fn", false, false, true, sdmk->t, sdmk->dm);
			ob->measure("fn", false, true, true, sdmk->t, sdmk->dm);
			ob->measure("sx", false, true, true, sdmk->t, sdmk->dm);
			ob->measure("sy", false, true, true, sdmk->t, sdmk->dm);
			ob->measure("sz", false, true, true, sdmk->t, sdmk->dm);
			if (pmp.active()) elight->probe(0, sdmk->t, sdmk->dm, sdmk->oneminusdm);
		}
	}

	void evolve_euler(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (ionode) printf("\n==================================================\n");
		if (ionode) printf("start density matrix evolution (euler method)\n");
		if (ionode) printf("==================================================\n");

		for (double it = 1; sdmk->t < sdmk->tend; it += 1)
			evolve_euler_one_step(it);
	}

	void evolve_gsl(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (ionode) printf("\n==================================================\n");
		if (ionode) printf("start density matrix evolution (rkf45 method from GSL)\n");
		if (ionode) printf("==================================================\n");

		// firstly do one step with tiny dt to get initial relaxation rate
		if (!param->restart) evolve_euler_one_step(1);

		//MPI_Barrier(MPI_COMM_WORLD);
		//return; // debug

		// set ODE solver
		size_t size_y = sdmk->nk_glob*(size_t)std::pow(sdmk->nb, 2) * 2;
		gsl_odeiv2_system sys = { func<Tl, Te, Telight, Teph>, NULL, size_y, this };
		gsl_odeiv2_driver* d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, ode.hstart, ode.epsabs, 0.0);
		gsl_odeiv2_driver_set_hmin(d, ode.hmin);
		if (pmp.active() && elight->during_pump(sdmk->t)) gsl_odeiv2_driver_set_hmax(d, ode.hmax_pump);
		else gsl_odeiv2_driver_set_hmax(d, ode.hmax);
		double y[size_y];
		copy_real_from_complex(y, sdmk->dm, size_y / 2);

		// evolution
		MPI_Barrier(MPI_COMM_WORLD);
		double ti = sdmk->t;
		for (int it = it0 + 1; sdmk->t < sdmk->tend; it += 1, ode.ncalls = 0){
			ti += dt_current(it);
			if (pmp.active()){
				if (elight->enter_pump(sdmk->t, ti)) gsl_odeiv2_driver_set_hmax(d, ode.hmax_pump);
				if (elight->leave_pump(sdmk->t, ti)) gsl_odeiv2_driver_set_hmax(d, ode.hmax);
			}

			int status = gsl_odeiv2_driver_apply(d, &sdmk->t, ti, y);
			if (status != GSL_SUCCESS) throw std::invalid_argument("!GSL_SUCCESS");
			report(it);
			if (ionode) printf("ncalls= %d at ti= %lg fs\n", ode.ncalls, ti / fs);
		}
		gsl_odeiv2_driver_free(d);
	}

	void evolve_euler_one_step(int it){
		compute(sdmk->t);
		sdmk->update_dm_euler(dt_current(it));
		sdmk->t += dt_current(it);
		report(it);
	}

	void compute_ob_dot(int it){
		compute(sdmk->t);
		report_dot(it);
	}

	double dt_current(int it){
		if (param->compute_dot_only)
			return std::min(ode.hstart, std::min(sdmk->dt / 10, sdmk->dt_pump / 10));
		else if (!param->restart && it == 1)
			return std::min(ode.hstart, std::min(sdmk->dt / 10, sdmk->dt_pump / 10));
		else if (!param->restart && it == 2)
			return pmp.active() && elight->during_pump(sdmk->t) ? elight->dt : sdmk->dt
			- std::min(ode.hstart, std::min(sdmk->dt / 10, sdmk->dt_pump / 10));
		else
			return pmp.active() && elight->during_pump(sdmk->t) ? elight->dt : sdmk->dt;
	}

	void compute(double t){
		ode.ncalls++;

		sdmk->set_oneminusdm(); // also zeros(ddmdt)

		if (pmp.active() && (elight->pumpMode == "lindblad" || elight->pumpMode == "coherent")){
			if (alg.ddmeq || alg.phenom_relax) sdmk->set_dm_eq(param->temperature, elec->e_dm, elec->nv_dm);
			if (alg.ddmeq) eph->compute_ddm_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
		}

		if (!alg.H0hasBS){ // coherent dynamics, including BS
			sdmk->evolve_coh(t, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}

		if (pmp.active() && elight->pumpMode != "perturb"){
			elight->evolve_pump(t, sdmk->dm, sdmk->oneminusdm, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}

		if (alg.eph_enable){
			eph->evolve_driver(t, sdmk->dm, sdmk->oneminusdm, sdmk->ddmdt_term);
			//eph->evolve_debug(sdmk->t, sdmk->dm, sdmk->oneminusdm, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}

		if (alg.phenom_relax){
			phnm_rlx->evolve_driver(t, sdmk->dm, sdmk->dm_eq, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}
	}

	void report(int it){
		if (it > it0) sdmk->write_dm_tofile(sdmk->t);
		bool print_ene = (it - it0) % ob->freq_measure_ene == 0;
		ob->measure("fn", false, true, print_ene, sdmk->t, sdmk->dm);
		ob->measure("sx", false, true, print_ene, sdmk->t, sdmk->dm);
		ob->measure("sy", false, true, print_ene, sdmk->t, sdmk->dm);
		ob->measure("sz", false, true, print_ene, sdmk->t, sdmk->dm);
		if (pmp.active()) elight->probe(it, sdmk->t, sdmk->dm, sdmk->oneminusdm);
	}

	void report_dot(int it){
		ob->measure("sx", true, true, false, sdmk->t, sdmk->ddmdt);
		ob->measure("sy", true, true, false, sdmk->t, sdmk->ddmdt);
		ob->measure("sz", true, true, false, sdmk->t, sdmk->ddmdt);
	}
};

// notice that in this code, memory of 2D array is continous
template<class Tl, class Te, class Telight, class Teph>
int func(double t, const double y[], double dydt[], void *params){
	//auto dmdyn = (dm_dynamics<lattice, electron, electronlight, electronphonon> *)params;
	auto dmdyn = (dm_dynamics<Tl, Te, Telight, Teph> *)params;
	size_t n = dmdyn->sdmk->nk_glob*(size_t)std::pow(dmdyn->sdmk->nb, 2);
	copy_complex_from_real(dmdyn->sdmk->dm, y, n);

	dmdyn->compute(t);

	copy_real_from_complex(dydt, dmdyn->sdmk->ddmdt, n);
	return GSL_SUCCESS;
}