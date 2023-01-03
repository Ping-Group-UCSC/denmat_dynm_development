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
#include "material_model.h"

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
	double *tau_neq;

	dm_dynamics(Tl* latt, parameters* param, Te* elec, Telight* elight, Teph* eph)
		: latt(latt), param(param), elec(elec), elight(elight), eph(eph)
	{
		// density matrix
		sdmk = new singdenmat_k(param, &mpk, elec); // k-independent single density matrix
		//if (alg.use_dmDP_in_evolution) sdmk->init_dmDP(elec->ddm_Bpert, elec->ddm_Bpert_neq);
		sdmk->init_Hcoh(elec->H_BS, elec->H_Ez, elec->e_dm);
		// probe ground state
		sdmk->init_dm(elec->f_dm);
		if (pmp.active()) elight->probe(-1, sdmk->t, sdmk->dm, sdmk->oneminusdm);
		// initialize density matrix with spin imbalance
		if (param->restart)
			sdmk->read_dm_restart();
		else{
			if (pmp.active() && pmp.laserAlg == "perturb"){
				if (ionode) printf("initial density matrix inbalance is induced by pump (Gaussian) pulse perturbation\n");
				sdmk->init_dm(elight->dm_pump);
			}
			else{
				if (param->Bpert.length() > 1e-10) { if (ionode) printf("initial density matrix inbalance induced by Bpert (1st order)\n"); }
				else { if (ionode) printf("no initial density matrix inbalance\n"); }
				sdmk->init_dm(elec->dm_Bpert);
				//if (!alg.use_dmDP_taufm_as_init) sdmk->init_dm(elec->dm_Bpert);
				//else sdmk->init_dm(elec->dm_Bpert_neq);
			}
		}
		if (!(pmp.active() && (pmp.laserAlg == "lindblad" || pmp.laserAlg == "coherent")) && alg.scatt_enable){
			if (alg.linearize || alg.ddmdteq || alg.phenom_relax) sdmk->set_dm_eq(param->temperature, elec->e_dm, elec->nv_dm);
			if (alg.linearize) eph->set_Lsc(sdmk->f_eq);
			if (alg.ddmdteq){
				if (!alg.linearize_dPee) eph->reset_scatt(true, true, sdmk->dm_eq, nullptr, sdmk->t, sdmk->f_eq);
				if (!alg.linearize_dPee) eph->compute_ddmdt_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
			}
		}

		// phenomenon 
		if (alg.phenom_relax) phnm_rlx = new phenom_relax(param, elec->nk, elec->nb_dm, sdmk->dm_eq);

		// observables
		ob = new ob_1dmk<Tl, Te>(latt, param, elec, eph->bStart, eph->bEnd);
		report(0, false, false, true); // report initial quatities: dos, occupation, spin, probe to stdout
		if (!param->restart) report(0); // write initial excess quantities in files

		// compute relaxtion according to intial density matrix
		if (param->compute_tau_only){ compute(sdmk->t); report_tau(0); } //compute initial relaxation time
		//if (!param->restart && param->compute_tau_only && alg.DP_beyond_carrierlifetime) compute_tauDP_beyond_carrierlifetime(); // compute DP relaxation time
	}
	/*
	void compute_tauDP_beyond_carrierlifetime(){
		if (!elec->imsig_eph_kn) return;
		MPI_Barrier(MPI_COMM_WORLD);
		if (ionode) printf("\n**************************************************\n");
		if (ionode) printf("Compute tau^neq and DP relaxation time:\n");

		tau_neq = new double[elec->nk]{0};
		for (int ik = 0; ik < elec->nk; ik++)
			tau_neq[ik] = 0.5 / elec->imsig_eph_k[ik];
		if (exists("ratio_tauneq_taufm.dat")){
			double *ratio = new double[elec->nk]{1.};
			string fname = "ratio_tauneq_taufm.dat"; FILE *filratio = fopen(fname.c_str(), "rb");
			fread(ratio, sizeof(double), elec->nk, filratio);
			fclose(filratio);
			for (int ik = 0; ik < elec->nk; ik++)
				tau_neq[ik] *= ratio[ik];
			if (ionode) elec->print_array_atk(ratio, "tau^neq / tau^FM:\n");

			double rate_DP = 0;
			for (int ik = 0; ik < elec->nk; ik++)
				rate_DP += tau_neq[ik] * elec->DP_precess_fac[ik];
			if (ionode) printf("tau_DP using tau^neq from real-time: %.3lf ps\n", 1 / rate_DP / ps);
		}

		double *sum_dfde_k = new double[elec->nk]{0}, sum_dfde = 0;
		complex **commut = alloc_array(elec->nk, elec->nb_dm*elec->nb_dm); // -i[e,ddm_eq] / sum_dfde_k
		for (int ik = 0; ik < elec->nk; ik++){
			commutator_mat_diag(commut[ik], elec->e_dm[ik], elec->ddm_Bpert[ik], elec->nb_dm, cmi);
			for (int i = 0; i < elec->nb_eph; i++){
				double dfde = elec->f[ik][i + elec->bStart_eph] * (elec->f[ik][i + elec->bStart_eph] - 1.) / param->temperature * param->Bpert.length();
				sum_dfde_k[ik] += dfde;
				sum_dfde += dfde;
			}
			for (int ibb = 0; ibb < elec->nb_dm*elec->nb_dm; ibb++)
				commut[ik][ibb] = fabs(sum_dfde_k[ik]) < 1e-30 ? 0 : commut[ik][ibb] / sum_dfde_k[ik];
		}
		complex *ddm_neq_enlarged = new complex[elec->nb_dm*elec->nb_dm];
		complex *ddmdt_enlarged = new complex[elec->nb_dm*elec->nb_dm];

		int niter = 200; double tau_DP_old = 0;
		for (int iter = 0; iter < niter; iter++){
			// ddm_neq_k = -i tau^neq_k [e_k, ddm_Bpert_k]
			for (int ik = 0; ik < elec->nk; ik++){
				commutator_mat_diag(elec->ddm_Bpert_neq[ik], elec->e_dm[ik], elec->ddm_Bpert[ik], elec->nb_dm, cmi);
				for (int ibb = 0; ibb < elec->nb_dm*elec->nb_dm; ibb++)
					elec->ddm_Bpert_neq[ik][ibb] *= tau_neq[ik];
			}
			// dm_neq = f + ddm_eq + ddm_neq
			trunc_copy_arraymat(elec->dm_Bpert_neq, elec->dm_Bpert, elec->nk, elec->nb_dm, 0, elec->nb_dm);
			axbyc(elec->dm_Bpert_neq, elec->ddm_Bpert_neq, elec->nk, elec->nb_dm*elec->nb_dm, c1, c1); // y=ax+by+c, void axbyc(complex **y, complex **x, int n1, int n2, complex a = c1, complex b = c0, complex c = c0);

			sdmk->init_dm(elec->dm_Bpert_neq);
			// directly compute relaxation time using dm_neq = f + ddm_eq + ddm_neq
			compute(sdmk->t); report_tau(0, "_EYDP");
			// compute tau^coincide
			double rate_coincide = -trace_AB(elec->ddm_Bpert, sdmk->ddmdt, elec->nk, elec->nb_dm) / elec->trace_sq_ddm_tot;
			if (ionode) printf("tot= %lg dot= %lg tau^coincide = %lg ps\n", elec->trace_sq_ddm_tot, trace_AB(elec->ddm_Bpert, sdmk->ddmdt, elec->nk, elec->nb_dm), 1 / rate_coincide / ps);

			// compute only the scattering part of ddmdt(ddm_neq) for tau_DP calculation
			compute(sdmk->t, false);
			report_tau(0, "_EY"); // report relaxation due to scattering

			// compute tau_neq and tau_DP
			double rate_DP = 0;
			double rate_neq_avg = 0; int count_negative = 0;
			for (int ik = 0; ik < elec->nk; ik++){
				if (fabs(sum_dfde_k[ik]) < 1e-30) continue;
				// enlarge ddm_neq and ddmdt to improve the numerical accuracy when computing trace of the matrix multiplication of two
				for (int ibb = 0; ibb < elec->nb_dm*elec->nb_dm; ibb++){
					ddm_neq_enlarged[ibb] = elec->ddm_Bpert_neq[ik][ibb] / sum_dfde_k[ik]; // enlarge ddm_eq when dfde is tiny
					ddmdt_enlarged[ibb] = sdmk->ddmdt[ik][ibb] / sum_dfde_k[ik];
				}
				double trace_sq_ddm_neq = trace_square_hermite(ddm_neq_enlarged, elec->nb_dm);
				double trace_ddmneq_ddmdt = trace_AB(ddm_neq_enlarged, ddmdt_enlarged, elec->nb_dm);
				double rate_new = -trace_ddmneq_ddmdt / trace_sq_ddm_neq - rate_coincide;

				if (elec->has_precess[ik]) tau_neq[ik] = alg.mix_tauneq / rate_new + (1 - alg.mix_tauneq)*tau_neq[ik];
				if (tau_neq < 0 && alg.positive_tauneq){ tau_neq = 0; }
				if (tau_neq <= 0){ count_negative++; }
				rate_DP += tau_neq[ik] * elec->DP_precess_fac[ik];
				rate_neq_avg += trace_ddmneq_ddmdt * sum_dfde_k[ik] * sum_dfde_k[ik];
			}
			rate_neq_avg = -rate_neq_avg / trace_square_hermite(elec->ddm_Bpert_neq, elec->nk, elec->nb_dm);

			if (ionode) printf("iter %d: tau^neq_avg= %lg fs  tau_DP= %.3lf ps\n", iter, 1 / rate_neq_avg / fs, 1 / rate_DP / ps);
			if (ionode && count_negative) printf("%d / %d = %lg%% of tau^neq are negtive\n", count_negative, elec->nk, (double)count_negative / (double)elec->nk);
			if (ionode && (iter % 10 == 0 || iter < 2)) elec->print_array_atk(tau_neq, "tau_neq:\n", fs);

			if (iter >= 2 && fabs(1 / rate_DP - tau_DP_old) < 0.001 * ps) break;
			tau_DP_old = 1 / rate_DP;
		}

		// set dm back
		sdmk->init_dm(elec->dm_Bpert);
		if (ionode) printf("\n**************************************************\n");
		MPI_Barrier(MPI_COMM_WORLD);
	}
	*/
	void evolve_euler(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (ionode) printf("\n==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("start density matrix evolution (euler method)\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");

		for (double it = 1; sdmk->t < sdmk->tend; it += 1, ode.ncalls = 0)
			evolve_euler_one_step(it);
	}

	void evolve_gsl(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (ionode) printf("\n==================================================\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("start density matrix evolution (rkf45 method from GSL)\n");
		if (ionode) printf("==================================================\n");
		if (ionode) printf("==================================================\n");

		// set ODE solver
		size_t size_y = sdmk->nk_glob*(size_t)std::pow(sdmk->nb, 2) * 2;
		gsl_odeiv2_system sys = { func<Tl, Te, Telight, Teph>, NULL, size_y, this };
		gsl_odeiv2_driver* d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, ode.hstart, ode.epsabs, 0.0);
		gsl_odeiv2_driver_set_hmin(d, ode.hmin);
		if (pmp.active() && elight->during_laser(sdmk->t)) gsl_odeiv2_driver_set_hmax(d, ode.hmax_laser);
		else gsl_odeiv2_driver_set_hmax(d, ode.hmax);
		double y[size_y];
		copy_real_from_complex(y, sdmk->dm, size_y / 2);

		// evolution
		MPI_Barrier(MPI_COMM_WORLD);
		double ti = sdmk->t;
		for (int it = 1; sdmk->t < sdmk->tend; it += 1, ode.ncalls = 0){
			if ((it-1) % ob->freq_compute_tau == 0){ compute(sdmk->t); report_tau(it); ode.ncalls = 0; } // notice that you need to call subroutine "compute" before "report_tau"
			ti += dt_current();
			if (pmp.active()){
				if (elight->enter_laser(sdmk->t, ti)) gsl_odeiv2_driver_set_hmax(d, ode.hmax_laser);
				if (elight->leave_laser(sdmk->t, ti)) gsl_odeiv2_driver_set_hmax(d, ode.hmax);
			}
			update_scatt_outside(sdmk->t, it);

			int status = gsl_odeiv2_driver_apply(d, &sdmk->t, ti, y);
			if (status != GSL_SUCCESS) throw std::invalid_argument("!GSL_SUCCESS");
			{ copy_complex_from_real(sdmk->dm, y, size_y / 2); report(it); } // ensure dm is at current time
			if (ionode) printf("ncalls= %d at ti= %lg fs\n", ode.ncalls, ti / fs);
		}
		gsl_odeiv2_driver_free(d);
	}

	void evolve_euler_one_step(int it){
		update_scatt_outside(sdmk->t, it);
		compute(sdmk->t);
		if ((it-1) % ob->freq_compute_tau == 0) report_tau(it);
		sdmk->update_dm_euler(dt_current());
		sdmk->t += dt_current();
		report(it);
	}

	double dt_tiny(){
		return std::min(ode.hstart, std::min(sdmk->dt / 10, sdmk->dt_laser / 10));
	}
	double dt_current(){
		return pmp.active() && elight->during_laser(sdmk->t) ? elight->dt : sdmk->dt;
	}

	void compute(double t, bool active_coh = true){
		if (exists("EXIT_DMD")){ { if (ionode) system("rm EXIT_DMD"); } mpkpair.mpi_abort("clean exit", 0); } // user can "touch EXIT_DMD to exit the program"
		sdmk->t = t; ode.ncalls++;
		if (ionode && ode.ncalls > 12 && ode.ncalls % 6 == 0) { printf("t= %lg fs\n", t / fs); fflush(stdout); }

		//if (alg.use_dmDP_in_evolution) sdmk->use_dmDP(elec->f_dm);
		if (alg.semiclassical) zeros_off_diag(sdmk->dm, sdmk->nk_glob, sdmk->nb);
		sdmk->set_oneminusdm(); // also zeros(ddmdt)

		if (active_coh && (alg.picture == "schrodinger" || elec->H_BS || elec->H_Ez)){ // coherent dynamics, including BS
			sdmk->evolve_coh(t, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}

		if (pmp.active() && pmp.laserAlg != "perturb" && elight->during_laser(t)){
			elight->evolve_laser(t, sdmk->dm, sdmk->oneminusdm, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}

		if (alg.scatt_enable){
			update_scatt_inside(sdmk->t);
			if (pmp.active() && pmp.laserAlg != "perturb" && elight->during_laser(t)){
				if (alg.ddmdteq || alg.phenom_relax || update_eimp_model_inside(t)) sdmk->set_dm_eq(param->temperature, elec->e_dm, elec->nv_dm);
				if (alg.ddmdteq) eph->compute_ddmdt_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
			}
			eph->evolve_driver(t, sdmk->dm, sdmk->oneminusdm, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}

		if (alg.phenom_relax){
			phnm_rlx->evolve_driver(t, sdmk->dm, sdmk->dm_eq, sdmk->ddmdt_term);
			sdmk->update_ddmdt(sdmk->ddmdt_term);
		}

		if (alg.semiclassical) zeros_off_diag(sdmk->ddmdt, sdmk->nk_glob, sdmk->nb);
	}

	void update_scatt_inside(double t){
		bool update_eimp = update_eimp_model_inside(t), update_ee = update_ee_model_inside();
		if (update_eimp){
			sdmk->set_dm_eq(param->temperature, elec->e_dm, elec->nv_dm);
			if (alg.ddmdteq && !alg.linearize_dPee) eph->reset_scatt(true, true, sdmk->dm_eq, nullptr, sdmk->t, sdmk->f_eq);
			if (alg.ddmdteq && !alg.linearize_dPee) eph->compute_ddmdt_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
		}
		eph->reset_scatt(update_eimp, update_ee, sdmk->dm, sdmk->oneminusdm, t, sdmk->f_eq);
		if (update_eimp && alg.linearize_dPee) eph->compute_ddmdt_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
	}
	void update_scatt_outside(double t, int it){
		bool update_eimp = update_eimp_model_outside(t, it), update_ee = update_ee_model_outside(it);
		if (update_eimp){
			sdmk->set_dm_eq(param->temperature, elec->e_dm, elec->nv_dm);
			if (alg.ddmdteq && !alg.linearize_dPee) eph->reset_scatt(true, true, sdmk->dm_eq, nullptr, sdmk->t, sdmk->f_eq);
			if (alg.ddmdteq && !alg.linearize_dPee) eph->compute_ddmdt_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
		}
		sdmk->set_oneminusdm(); // also zeros(ddmdt)
		eph->reset_scatt(update_eimp, update_ee, sdmk->dm, sdmk->oneminusdm, t, sdmk->f_eq);
		if (update_eimp && alg.linearize_dPee) eph->compute_ddmdt_eq(sdmk->f_eq); // compute time derivative of density matrix in equilibrium
	}
	bool update_eimp_model_inside(double t){
		bool update = pmp.active() && elight->during_laser(t) && param->freq_update_eimp_model < 0 && !alg.linearize;
		if (!update) return false;
		for (int iD = 0; iD < eip.ni.size(); iD++)
			if (eph->eimp[iD]->eimp_model != nullptr && update) return true;
		return false;
	}
	bool update_ee_model_inside(){
		return eph->ee_model != nullptr && eep.eeMode == "Pee_update" && param->freq_update_ee_model < 0 && !alg.linearize;
	}
	bool update_eimp_model_outside(double t, int it){
		bool update = pmp.active() && elight->during_laser(t) && param->freq_update_eimp_model > 0 && (it - 1) % param->freq_update_eimp_model == 0 && !alg.linearize;
		if (!update) return false;
		for (int iD = 0; iD < eip.ni.size(); iD++)
			if (eph->eimp[iD]->eimp_model != nullptr && update) return true;
		return false;
	}
	bool update_ee_model_outside(int it){
		return eph->ee_model != nullptr && eep.eeMode == "Pee_update" && param->freq_update_ee_model > 0 && (it - 1) % param->freq_update_ee_model == 0 && !alg.linearize;
	}

	void report(int it, bool diff = true, bool prtprobe = true, bool prtdos = false, string lable = ""){
		if (it % ob->freq_measure != 0) return;
		if (it > 0) sdmk->write_dm_tofile(sdmk->t);
		bool print_ene = it % ob->freq_measure_ene == 0;
		if (prtdos) ob->measure("dos", lable, true, true, sdmk->t, sdmk->dm); // for dos, diff == true just means file name has no "initial"
		ob->measure("fn", lable, diff, print_ene, sdmk->t, sdmk->dm);
		if (ob->print_layer_occ) ob->measure("layer", lable, diff, false, sdmk->t, sdmk->dm);
		if (ob->print_layer_spin) ob->measure("layerspin", lable, diff, false, sdmk->t, sdmk->dm);
		ob->measure("sx", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		ob->measure("sy", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		ob->measure("sz", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		if (ob->needL) ob->measure("lx", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		if (ob->needL) ob->measure("ly", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		if (ob->needL) ob->measure("lz", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		ob->measure("jx", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		ob->measure("jy", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		ob->measure("jz", lable, diff, print_ene && diff, sdmk->t, sdmk->dm);
		ob->measure("s-t2-wu", lable, diff, false, sdmk->t, sdmk->dm);
		ob->measure("s-t2star", lable, diff, false, sdmk->t, sdmk->dm);
		ob->measure("s-t2-mani", lable, diff, false, sdmk->t, sdmk->dm);
		ob->measure("entropy_bloch", lable, diff, false, sdmk->t, sdmk->dm);
		ob->measure("entropy_vN", lable, diff, false, sdmk->t, sdmk->dm);
		if (prtprobe && pmp.active()) elight->probe(it, sdmk->t, sdmk->dm, sdmk->oneminusdm);
	}

	void report_tau(int it, string lable = ""){
		bool print_ene = param->compute_tau_only || (it-1) % ob->freq_measure_ene == 0;
		//ob->measure("fn", lable, true, print_ene, sdmk->t, sdmk->dm, sdmk->ddmdt, dt_tiny());
		if (!(param->Bpert.length() > 1e-10 && param->Bpert[0] == 0)) ob->measure("sx", lable, true, print_ene, sdmk->t, sdmk->dm, sdmk->ddmdt, dt_tiny());
		if (!(param->Bpert.length() > 1e-10 && param->Bpert[1] == 0)) ob->measure("sy", lable, true, print_ene, sdmk->t, sdmk->dm, sdmk->ddmdt, dt_tiny());
		if (!(param->Bpert.length() > 1e-10 && param->Bpert[2] == 0)) ob->measure("sz", lable, true, print_ene, sdmk->t, sdmk->dm, sdmk->ddmdt, dt_tiny());
		//if (param->print_along_kpath) ob->ddmk_ana_drive((it - 1) / ob->freq_compute_tau + 1, sdmk->t, sdmk->dm);
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