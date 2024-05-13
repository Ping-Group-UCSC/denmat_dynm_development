#include <InputMap.h>
#include <FeynWann.h>
#include <core/Units.h>
#include <lindblad/Lindblad.h>


//! Print a complex vector with real and imaginary part separated (used for polarizations)
inline void print(FILE* fp, const vector3<complex>& v, const char* format="%lg ")
{	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].real()); std::fprintf(fp, "] + 1j*");
	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].imag()); std::fprintf(fp, "]\n");
}

//! Normalize a complex vector (used to handle polarizations)
inline vector3<complex> normalize(const vector3<complex>& v)
{	return v * (1./sqrt(v[0].norm() + v[1].norm() + v[2].norm()));
}


int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Lindblad dynamics in an ab initio Wannier basis");

	//Get input parameters:
	InputMap inputMap(ip.inputFilename);
	LindbladParams lp;
	//--- doping / temperature
	lp.dmu = inputMap.get("dmu", 0.) * eV; //optional: shift in fermi level from neutral value / VBM in eV (default: 0)
	lp.T = inputMap.get("T") * Kelvin; //temperature in Kelvin (ambient phonon T = initial electron T)
	//--- spectrum vs real-time dynamics (and linear vs nonlinear)
	const string mode = inputMap.getString("mode"); //RealTime or Spectrum
	if((mode!="RealTime") and (mode!="Spectrum"))
		die("\nmode must be 'RealTime' or 'Spectrum'\n");
	lp.spectrumMode = (mode == "Spectrum");
	lp.linearized = inputMap.getBool("linearized", false);
	//--- check compilation flags for mode
	#ifndef SCALAPACK_ENABLED
	if(lp.spectrumMode)
		die("\nSpectrum (dense diagonalization) mode requires linking with ScaLAPACK.\n");
	#endif
	#ifndef PETSC_ENABLED
	if(lp.linearized and (not lp.spectrumMode))
		die("\nRealTime (linearized time evolution) mode requires linking with PETSc.\n");
	#endif
	//--- spectrum diagonalization / time evolution parameters
	if(lp.spectrumMode)
	{	lp.blockSize = int(inputMap.get("blockSize", 64));
		const string diagMethodName = inputMap.has("diagMethod") ? inputMap.getString("diagMethod") : "PDHSEQRm";
		#ifdef SCALAPACK_ENABLED
		if(not BlockCyclicMatrix::diagMethodMap.getEnum(diagMethodName.c_str(), lp.diagMethod))
			die("diagMethod must be one of %s\n", BlockCyclicMatrix::diagMethodMap.optionList().c_str());
		#endif
	}
	else
	{	lp.dt = inputMap.get("dt") * fs; //time interval between reports
		lp.tStop = inputMap.get("tStop") * fs; //stopping time for simulation
		lp.tStep = inputMap.get("tStep", 0.) * fs; //if non-zero, time step for fixed-step (non-adaptive) integrator
		lp.tolAdaptive = inputMap.get("tolAdaptive", 1e-3); //relative tolerance for adaptive integrator
	}
	//--- pump
	const string pumpMode = lp.spectrumMode
		? "Bfield" //only uses pumpB to set perturbation strength for spectrum mode
		: inputMap.getString("pumpMode"); //must be Evolve, Perturb or Bfield
	if(pumpMode!="Evolve" and pumpMode!="Perturb" and pumpMode!="Bfield")
		die("\npumpMode must be 'Evolve' or 'Perturb' or 'Bfield'\n");
	lp.pumpEvolve = (pumpMode == "Evolve");
	lp.pumpBfield = (pumpMode == "Bfield");
	lp.pumpB = inputMap.getVector("pumpB", vector3<>()) * Tesla; //perturbing initial magnetic field in Tesla (used only in Bfield mode)
	lp.pumpOmega = inputMap.get("pumpOmega", pumpMode=="Bfield" ? 0. : NAN) * eV; //pump frequency in eV (used only in Evolve/Perturb modes)
	lp.pumpTau = inputMap.get("pumpTau", pumpMode=="Bfield" ? 0. : NAN)*fs; //Gaussian pump pulse width in fs, or 0 for CW
	lp.pumpI = inputMap.get("pumpI", pumpMode=="Bfield" ? 0. : NAN)
		* (lp.pumpTau
			? Joule / std::pow(meter, 2) //pump integrated energy flux in J/m^2 (gaussian pulse)
			: Joule / (std::pow(meter, 2) * sec) //pump intensity in W/m^2 (CW)
		); 
	lp.pumpSigma = inputMap.get("pumpSigma",
		pumpMode=="Bfield"
		? 0.
		: (lp.pumpTau ? (sqrt(0.5)/lp.pumpTau)/eV : NAN) //default from tau for Gauss pulse
	)*eV; //Energy-conservation width for pump in eV
	lp.pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes
	while(true)
	{	int iPol = int(lp.pol.size())+1;
		ostringstream oss; oss << iPol;
		string polName = oss.str();
		vector3<> polRe = inputMap.getVector("polRe"+polName, vector3<>(0.,0.,0.)); //Real part of polarization
		vector3<> polIm = inputMap.getVector("polIm"+polName, vector3<>(0.,0.,0.)); //Imag part of polarization
		if(polRe.length_squared() + polIm.length_squared() == 0.) break; //End of probe polarizations
		lp.pol.push_back(normalize(complex(1,0)*polRe + complex(0,1)*polIm));
	}
	lp.omegaMin = inputMap.get("omegaMin", lp.pol.size() ? NAN : 0.) * eV; //start of frequency grid for probe response
	lp.omegaMax = inputMap.get("omegaMax", lp.pol.size() ? NAN : 0.) * eV; //end of frequency grid for probe response
	lp.domega = inputMap.get("domega", lp.pol.size() ? NAN : 0.) * eV; //frequency resolution for probe calculation
	lp.tau = inputMap.get("tau", lp.pol.size() ? NAN : 0.) * fs; //Gaussian probe pulse width (sigma of amplitude) in fs
	//--- general options
	lp.Bext = inputMap.getVector("Bext", vector3<>()) * Tesla; //constant external magnetic field post-initialization in Tesla
	lp.orbitalZeeman = inputMap.getBool("orbitalZeeman", false); //whether to include L.B coupling with orbital angular momentum
	lp.spinEchoB = inputMap.getVector("spinEchoB", vector3<>()) * Tesla; //spin-echo perturbing field in Tesla
	lp.spinEchoDelay = inputMap.get("spinEchoDelay", 0.) * fs; //spin-echo delay time in fs
	lp.spinEchoOmega = inputMap.get("spinEchoOmega", 0.) * eV; //spin-echo Larmor frequency (x hbar) in eV (if zero, set based on Bext)
	lp.dE = inputMap.get("dE") * eV; //energy resolution for distribution functions
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	if(ePhMode!="Off" and ePhMode!="DiagK")
		die("\nePhMode must be 'Off' or 'DiagK'\n");
	lp.ePhEnabled = (ePhMode != "Off");
	if(lp.spectrumMode and (not lp.ePhEnabled))
		die("\nePhMode must be 'DiagK' in Spectrum mode\n");
	lp.defectFraction = inputMap.get("defectFraction", 0.); //fractional concentration of defects if any
	lp.verbose = inputMap.getBool("verbose", false);
	lp.saveDist = inputMap.getBool("saveDist", true);
	lp.inFile = inputMap.has("inFile") ? inputMap.getString("inFile") : "ldbd.dat"; //input file name
	lp.checkpointFile = inputMap.has("checkpointFile") ? inputMap.getString("checkpointFile") : ""; //checkpoint file name
	lp.evecFile = inputMap.has("evecFile") ? inputMap.getString("evecFile") : "ldbd.evecs"; //eigenvector file name
	const string valleyModeStr = inputMap.has("valleyMode") ? inputMap.getString("valleyMode") : "None";
	EnumStringMap<ValleyMode> valleyModeMap(ValleyNone, "None", ValleyInter, "Inter", ValleyIntra, "Intra");
	if(not valleyModeMap.getEnum(valleyModeStr.c_str(), lp.valleyMode))
		die("\nvalleyMode must be 'None' or 'Intra' or 'Inter'\n");
	lp.initialize(); //update derived parameters

	//Report converted input parameters as a check:
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("dmu = %lg\n", lp.dmu);
	logPrintf("T = %lg\n", lp.T);
	logPrintf("mode = %s\n", lp.spectrumMode ? "Spectrum" : "RealTime");
	logPrintf("linearized = %s\n", lp.linearized ? "yes" : "no");
	if(lp.spectrumMode)
	{	logPrintf("blockSize = %d\n", lp.blockSize);
		#ifdef SCALAPACK_ENABLED
		logPrintf("diagMethod = %s\n", BlockCyclicMatrix::diagMethodMap.getString(lp.diagMethod));
		#endif
	}
	else
	{	logPrintf("dt = %lg\n", lp.dt);
		logPrintf("tStop = %lg\n", lp.tStop);
		logPrintf("tStep = %lg\n", lp.tStep);
		logPrintf("tolAdaptive = %lg\n", lp.tolAdaptive);
	}
	logPrintf("pumpMode = %s\n", pumpMode.c_str());
	if(lp.pumpBfield)
	{	logPrintf("pumpB = "); lp.pumpB.print(globalLog, " %lg ");
	}
	else
	{	logPrintf("pumpOmega = %lg\n", lp.pumpOmega);
		logPrintf("pumpTau = %lg\n", lp.pumpTau);
		logPrintf("pumpI = %lg\n", lp.pumpI);
		logPrintf("pumpSigma = %lg\n", lp.pumpSigma);
		logPrintf("pumpPol = "); print(globalLog, lp.pumpPol);
	}
	if(lp.pol.size())
	{	for(int iPol=0; iPol<int(lp.pol.size()); iPol++)
		{	logPrintf("pol%d = ", iPol+1);
			print(globalLog, lp.pol[iPol]);
		}
		logPrintf("omegaMin = %lg\n", lp.omegaMin);
		logPrintf("omegaMax = %lg\n", lp.omegaMax);
		logPrintf("domega = %lg\n", lp.domega);
		logPrintf("tau = %lg\n", lp.tau);
	}
	logPrintf("Bext = "); lp.Bext.print(globalLog, " %lg ");
	logPrintf("orbitalZeeman = %s\n", lp.orbitalZeeman ? "yes" : "no");
	if(lp.spinEchoB.length_squared())
	{	logPrintf("spinEchoB = \n"); lp.spinEchoB.print(globalLog, " %lg ");
		logPrintf("spinEchoDelay = %lg\n", lp.spinEchoDelay);
		logPrintf("spinEchoOmega = %lg (may be overrided later)\n", lp.spinEchoOmega);
		logPrintf("spinEchoFlipTime = %lg (calculated later)\n", lp.spinEchoFlipTime);
	}
	logPrintf("dE = %lg\n", lp.dE);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("defectFraction = %lg\n", lp.defectFraction);
	logPrintf("verbose = %s\n", lp.verbose ? "yes" : "no");
	logPrintf("saveDist = %s\n", lp.saveDist ? "yes" : "no");
	logPrintf("inFile = %s\n", lp.inFile.c_str());
	logPrintf("checkpointFile = %s\n", lp.checkpointFile.c_str());
	logPrintf("evecFile = %s\n", lp.evecFile.c_str());
	logPrintf("valleyMode = %s\n", valleyModeMap.getString(lp.valleyMode));
	logPrintf("\n");

	//Initialize appropriate calculator class:
	std::shared_ptr<Lindblad> lbl;
	if(lp.spectrumMode)
		lbl = std::make_shared<LindbladSpectrum>(lp);
	else
	{	if(lp.linearized)
			lbl = std::make_shared<LindbladLinear>(lp);
		else
			lbl = std::make_shared<LindbladNonlinear>(lp);
	}

	//End of dry-run:
	logPrintf("Initialization completed successfully at t[s]: %9.2lf\n\n", clock_sec());
	logFlush();
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		lbl = 0;
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Perform requestec actions (dynamics/spectrum):
	lbl->calculate();
	
	//Cleanup:
	lbl = 0;
	FeynWann::finalize();
	return 0;
}
