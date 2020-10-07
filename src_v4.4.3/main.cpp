#include "DensityMatrix_Dynamics.h"
bool DEBUG = false;
string dir_debug = "debug_info/";
bool ionode = false;
algorithm alg;
string code = "";
ODEparameters ode;

void dm_dynamics_jdftx(parameters* param);
void dm_dynamics_mos2(parameters* param);

int main(int argc, char **argv)
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now(), t2;

	// mpi
	mpkpair.mpi_init(); // kpair parallel
	mpkpair2.mpi_init(); // kpair parallel
	mpk.mpi_init(); // k parallel
	ionode = mpkpair.ionode;

	// read parameters
	parameters* param = new parameters();
	param->read_param();

	if (code == "jdftx")
		dm_dynamics_jdftx(param);
	else if (code == "mos2")
		dm_dynamics_mos2(param);
	else if (code == "gaas")
		error_message("code == gaas not yet implemented");
	else
		error_message("the value of parameter code is not allowed");

	MPI_Barrier(MPI_COMM_WORLD);
	t2 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t2 - t1).count();
	if (ionode) cout << "total time: " << duration / 1.0e6 << " seconds" << endl;
	return 0;
}

void dm_dynamics_jdftx(parameters* param){
	//lattice
	lattice* latt = new lattice(param);
	latt->printLattice();

	// electron
	electron* elec = new electron(&mpk, latt, param);
	mpk.distribute_var("dm_dynamics_driver", elec->nk);
	param->mu = elec->set_mu_and_n(param->carrier_density); // if carrier_density is set (non-zero), recompute mu
	elec->compute_dm_Bpert_1st(param->Bpert, param->t0);
	if (elec->B.length() > 1e-10) elec->set_H_BS(mpk.varstart, mpk.varend);
	if ((alg.picture == "schrodinger" || param->t0 == 0) && param->Bpert.length() > 1e-12) elec->compute_DP_related(param->Bpert);
	// phonon
	phonon* ph = new phonon(latt, param);

	// electron-light or pump
	electronlight* elight;
	if (pmp.active()){
		elight = new electronlight(latt, param, elec, &mpk);
		if (param->pumpMode == "perturb") elight->pump_pert();
	}

	// electron-phonon
	electronphonon* eph = new electronphonon(&mpkpair, latt, param, elec, ph, alg.eph_sepr_eh, !alg.eph_need_elec);
	mpkpair.distribute_var("dm_dynamics_jdftx", eph->nkpair_glob);
	eph->set_eph();

	MPI_Barrier(MPI_COMM_WORLD);
	dm_dynamics<lattice, electron, electronlight, electronphonon>* dmdyn =
		new dm_dynamics<lattice, electron, electronlight, electronphonon>(latt, param, elec, elight, eph);

	//==================================================
	// evolve density matrix
	//==================================================
	if (param->compute_tau_only) return;
	if (alg.ode_method == "rkf45")
		dmdyn->evolve_gsl();
	else if (alg.ode_method == "euler")
		dmdyn->evolve_euler();
}

void dm_dynamics_mos2(parameters* param){
	//lattice
	lattice_MoS2* latt = new lattice_MoS2(param);
	latt->printLattice();

	// electron
	elec_model_MoS2* elec = new elec_model_MoS2(&mpk, latt, param);
	mpk.distribute_var("dm_dynamics_driver", elec->nk);
	if (param->B.length() > 1e-10) elec->set_H_BS(mpk.varstart, mpk.varend);
	// phonon
	ph_model_MoS2* ph = new ph_model_MoS2(latt, param);

	// electron-phonon
	eph_model_MoS2* eph = new eph_model_MoS2(&mpkpair, latt, param, elec, ph, alg.eph_sepr_eh, !alg.eph_need_elec);
	mpkpair.distribute_var("dm_dynamics_mos2", eph->nkpair_glob);
	eph->set_eph();

	electronlight_dummy* elight;

	MPI_Barrier(MPI_COMM_WORLD);
	dm_dynamics<lattice_MoS2, elec_model_MoS2, electronlight_dummy, eph_model_MoS2>* dmdyn =
		new dm_dynamics<lattice_MoS2, elec_model_MoS2, electronlight_dummy, eph_model_MoS2>(latt, param, elec, elight, eph);

	//==================================================
	// evolve density matrix
	//==================================================
	if (!param->compute_tau_only) return;
	if (alg.ode_method == "rkf45")
		dmdyn->evolve_gsl();
	else if (alg.ode_method == "euler")
		dmdyn->evolve_euler();
}