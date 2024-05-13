#include "common_headers.h"
#include "parameters.h"
#include "electron.h"
#include "phonon.h"
#include "ElectronPhonon.h"
#include "ElectronImpurity.h"

bool DEBUG = false;
string dir_debug = "debug_info/";
FILE *fpd = nullptr;
string dir_ldbd = "ldbd_data/";
string shole = "";

int main(int argc, char** argv){
	InitParams ip = FeynWann::initialize(argc, argv, "Initialize electron-phonon matrices for Lindblad dynamics");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	
	//Read input parameters
	parameters* param = new parameters();
	param->read_params(inputMap);

	//Initialize FeynWann:
	FeynWannParams fwp(&inputMap);	fwp.printParams(); // Bext, EzExt and scissor
	fwp.needVelocity = true;
	fwp.needSpin = true;
	fwp.needL = param->needL;
	fwp.needLayer = param->layerOcc;
	fwp.needDefect = param->defectName;
	fwp.needPhonons = (param->needOmegaPhMax || param->ePhEnabled); // should be always true, otherwise we have to be careful about the use of eLoop
	//fwp.maskOptimize = true;
	if (fileSize((fwp.wannierPrefix + ".mlwfImSigma_ePh").c_str()) > 0) fwp.needLinewidth_ePh = true;
	FeynWann fw(fwp);
	fw.omegaPhCut = param->ePhOmegaCut;

	//General q offsets and search maximum phonon energy
	phonon *ph = new phonon(fw, param);

	if (ip.dryRun){
		logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	if (mpiWorld->isHead()) system("mkdir ldbd_data");
	if (DEBUG && mpiWorld->isHead()) system("mkdir debug_info");

	//*****************************************************************************
	// Main parts of lindblad initionalization
	//*****************************************************************************
	//First pass (e only): select k-points and output electronic quantities
	lattice* latt = new lattice(fw, param);
	electron* elec = new electron(fw, latt, param);
	fw.energyOnly = true; // only energies are needed in kpointSelect
	elec->kpointSelect(elec->k0); // select energy range, bands and k points
	if (!fw.isMetal && param->assumeMetal_scatt) elec->kpointSelect_scatt();
	fw.energyOnly = false; // we need more than energies now
	elec->savekData(param, 0, ph->omegaMax); // output electronic quantities

	//Second pass (ph only): select and output k pairs
	ElectronPhonon *eph = new ElectronPhonon(fw, latt, param, elec, ph);
	if (param->ePhEnabled || param->nEphDelta > 999){
		eph->kpairSelect(ph->q0);
		eph->savekpairData();
	}

	//Third pass: output e-ph quantities and compute spin relaxation time using single-rate formulas
	if (param->ePhEnabled){
		if (!param->analyse_g2_E1E2fix){
			eph->compute_eph(); // save scattering matrices

			//Compute spin relaxation time using single-rate formulas and merge the parallel output files
			eph->relax_1step_useP();
			#ifdef MPI_SAFE_WRITE
			eph->merge_eph_P(); // This is safer for mpi output
			#else
			eph->merge_eph_P_mpi();
			#endif
			eph->relax_rate_usegm();
			eph->merge_eph_gm();
		}
		else eph->compute_eph_analyse_g2_E1E2fix();
	}

	// Fourth pass:  output e-i quantities and compute spin relaxation time using single-rate formulas
	ElectronImpurity *eimp = new ElectronImpurity(fw, latt, param, elec, ph);
	if (fw.fwp.needDefect.length()){
		eimp->kpairSelect();
		eimp->savekpairData();

		eimp->compute_eimp(); // save scattering matrices

		//Compute spin relaxation time using single-rate formulas and merge the parallel output files
		eimp->relax_1step_useP();
		#ifdef MPI_SAFE_WRITE
		eimp->merge_eimp_P(); // This is safer for mpi output
		#else
		eimp->merge_eimp_P_mpi();
		#endif
		eimp->relax_rate_useg();
		eimp->merge_eimp_g();
	}

	//*****************************************************************************

	//Cleanup:
	MPI_Barrier(MPI_COMM_WORLD);
	fw.free();
	FeynWann::finalize();
	return 0;
}