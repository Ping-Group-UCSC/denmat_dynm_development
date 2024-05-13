#include <core/Units.h>
#include "FeynWann.h"
#include "InputMap.h"
#include "Interp1.h"
#include "Histogram.h"
#include "Integrator.h"
#include <core/Util.h>
#include <core/Operators.h>
#include <core/matrix.h>
#include <fstream>

//Smooth v with a nearest-neighbor filter n times
void smooth(std::vector<double>& v, int n)
{	std::vector<double> vOut(v);
	for(int repeat=0; repeat<n; repeat++)
	{	for(size_t i=1; i<v.size()-1; i++)
			vOut[i] = 0.5 * v[i] + 0.25 * (v[i-1] + v[i+1]);
		v = vOut;
	}
}


struct ePhRelax : public Integrator<diagMatrix>
{
	Interp1 dos, dosPh;
	diagMatrix f0, dfPert; //Initial Fermi and perturbation due to photons
	double dt, tMax; //time step for output and max time
	double Z, detR; //electrons and volume per unit cell
	double T, dos0; //initial temperature and density of states at the Fermi level
	double minEcut, maxEcut; //E cutoff in eV below/above which holes/electrons can be injected
	double pInject; //probability that a carrier gets injected to substrate
	double De, scaledDe; //De, and De scaled by g(eF)**-3
	double pumpFWHM; //Pump full-width at half maximum
	bool pumpCW; //If true, CW excitation: ignore pumpFWHM and treat Uabs as absorbed power density
	bool fixedTl; //If true, hold lattice temperature fixed
	double scatterFactor; //Artificially increase the electron-phonon and electron-electron scattering rate by this value
	diagMatrix hInt; //energy resolved electron-phonon coupling
	diagMatrix Mee; //energy resolved electron-electron matrix element
	diagMatrix gM; //elementwise dos * Mee
	int eeStride; //coarse graining stride used to accelerate e-e calculation
	string runName;
	
	double Ee0, El0; //initial electronic and lattice energies
	std::vector<double> tArr; //time-points for which results are stored
	std::vector<diagMatrix> fArr; //distributions (and Tl) for each t in tArr
	
	//Energy grid:
	int nE; double Emin, dE;
	inline double Egrid(int i) const { return dos.xGrid[i]; }
	int ieMin, ieMax; //min and max active energy grid indices (that evolve with time)
	
	//Coarse grid for ee-scattering
	int nEcoarse; //length of coarse grid on active energy range
	double dEcoarse; //coarse grid spacing
	int ieStart, ieStop; //min and max coarse energy grid indices to deal with on current MPI process
	std::vector<double> kernel; //coarse-graining and interpolation kernel (length: 2*eeStride-1)
	//--- Gather from fine to coarse grid: set results
	inline void gather(const double* fine, double* coarse)
	{	double kernelSumInv = 1./eeStride;
		for(int i=0; i<nEcoarse; i++)
		{	double sum = 0.;
			for(int j=0; j<int(kernel.size()); j++)
				sum += kernel[j] * fine[ieMin+i*eeStride+j];
			coarse[i] = kernelSumInv * sum;
		}
	}
	//--- Scatter (interpolate) from coarse to fine grid: accumulate results
	inline void scatter(const double* coarse, double* fine)
	{	for(int i=0; i<nEcoarse; i++)
			for(int j=0; j<int(kernel.size()); j++)
				fine[ieMin+i*eeStride+j] += kernel[j] * coarse[i];
	}
	diagMatrix Mc, gMc; //coarse-grained versions of Mee and gM
	
	ePhRelax(int argc, char** argv)
	{
		//Parse the command line:
		
		InitParams ip = FeynWann::initialize(argc, argv, "Electron-phonon relaxation using Boltzmann equation");

		//Get the system parameters (mu, T, lattice vectors etc.)
		InputMap inputMap(ip.inputFilename);
		dt = inputMap.get("dt") * fs; //output time step in fs
		tMax = inputMap.get("tMax") * fs; //max time in fs
		Z = inputMap.get("Z"); //number of electrons per unit cell
		T = inputMap.get("T") * Kelvin; //initial temperature in Kelvin (electron and lattice)
		minEcut = inputMap.get("minEcut", -DBL_MAX)*eV; // energy cutoff below which holes can be injected
		maxEcut = inputMap.get("maxEcut", +DBL_MAX)*eV; //energy cutoff above which electrons can be injected
		pInject = inputMap.get("pInject", 0.); //probability of injection for carriers outside (minEcut, maxEcut)
		De = inputMap.get("De") / eV; //quadratic e-e lifetime coefficient in eV^-1
		fixedTl = inputMap.getBool("fixedTl", false); //If true, hold Tl fixed (an infinite thermal conductance contact)
		pumpCW = inputMap.getBool("pumpCW", false); //If true, pump is CW: ignore pumpFWHM and interpret Uabs as absorbed power density in Watt/meter^3
		pumpFWHM = inputMap.get("pumpFWHM", 0.) * fs; //Gaussian pump pulse width in fs (default: 0 => treat pump as instantaneous)
		double Uabs_unit = Joule/std::pow(meter,3); //Uabs is energy density for Gaussian pulses,
		if(pumpCW) Uabs_unit *= 1/sec; // ... and power density for CW
		const double Uabs = inputMap.get("Uabs") * Uabs_unit; //absorbed energy or power density in J/m^3 or W/m^3
		const double Eplasmon = inputMap.get("Eplasmon") * eV; //incident photon energy in eV
		scatterFactor= inputMap.get("scatterFactor", 1.); //Increase the e-e and e-ph scattering rate by this factor (dafeult: 1 => no scaling)
		const int nSmooth = (int)inputMap.get("nSmooth", 0.); //smoothing of energy-functions to stabilize dynamics (default: off)
		const string MeeFile = inputMap.getString("MeeFile"); //energy-dependent matrix element filename (use None to disable)
		eeStride = (int)inputMap.get("eeStride", 1.); //coarse graining stride used to accelerate e-e calculation
		runName = inputMap.getString("runName"); //prefix to use for output files
		const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
		detR = fabs(det(R));
		
		logPrintf("\nInputs after conversion to atomic units:\n");
		logPrintf("dt = %lg\n", dt);
		logPrintf("tMax = %lg\n", tMax);
		logPrintf("Z = %lg\n", Z);
		logPrintf("T = %lg\n", T);
		logPrintf("(minEcut,maxEcut) = (%lg, %lg)\n", minEcut, maxEcut);
		logPrintf("pInject: %lg\n", pInject);
		logPrintf("De = %lg\n", De);
		logPrintf("fixedTl = %s\n", (fixedTl ? "yes" : "no"));
		logPrintf("pumpCW = %s\n", (pumpCW ? "yes" : "no"));
		logPrintf("pumpFWHM = %lg\n", pumpFWHM);
		logPrintf("Uabs = %lg\n", Uabs);
		logPrintf("Eplasmon = %lg\n", Eplasmon);
		logPrintf("scatterFactor = %lg\n", scatterFactor);
		logPrintf("nSmooth = %d\n", nSmooth);
		logPrintf("MeeFile = %s\n", MeeFile.c_str());
		logPrintf("eeStride = %d\n", eeStride);
		logPrintf("runName = %s\n", runName.c_str());
		logPrintf("R:\n"); R.print(globalLog, " %lg ");
		logPrintf("detR = %lg\n", detR);
		
		//Read electron and phonon DOS (and convert to atomic units and per-unit volume):
		dos.init("eDOS.dat", eV, 1./(detR*eV));
		smooth(dos.yGrid[0], nSmooth);
		dosPh.init("phDOS.dat", eV, 1./(detR*eV));
		nE = dos.xGrid.size();
		dE = dos.dx;
		f0.resize(nE);
		
		//Read energy dependent matrix elements if specified:
		Mee.assign(nE, 1.); //default: no scaling relative to De
		if(MeeFile != "None")
		{	Interp1 MeeInterp;
			MeeInterp.init(MeeFile.c_str(), eV, 1.);
			for(int ie=0; ie<nE; ie++)
				Mee[ie] = MeeInterp(Egrid(ie));
		}
		gM.resize(nE);
		for(int ie=0; ie<nE; ie++)
			gM[ie] = dos.yGrid[0][ie] * Mee[ie];
		
		//Calculate maximum electron temperature after absorption (asymptote without e-ph):
		double Umax = get_Uthermal(T) + Uabs;
		//--- find bracketing interval:
		double Tmin = T, deltaT = 100*Kelvin;
		double Tmax = T + deltaT;
		while(get_Uthermal(Tmax) < Umax)
		{	Tmin = Tmax;
			Tmax = Tmin + deltaT;
		}
		//--- bisect:
		const double tol = 1e-2*Kelvin;
		double Tmid = 0.5*(Tmin+Tmax);
		while(Tmax-Tmin > tol)
		{	double Umid = get_Uthermal(Tmid);
			((Umid>Umax) ? Tmax : Tmin) = Tmid;
			Tmid = 0.5*(Tmin + Tmax);
		}
		logPrintf("Asymptotic electron temperature without e-ph, TeMax = %.2lf K\n", Tmid/Kelvin);
		
		//Determine initial Fermi distribution:
		double dmu = get_dmu(T);
		logPrintf("Initial Fermi distribution: dmu = %le eV\n", dmu/eV);
		//--- calculate density of states at the Fermi level:
		dos0 = 0.;
		for(int ie=0; ie<nE; ie++)
			dos0 += dE * dos.yGrid[0][ie] * fermiPrime((Egrid(ie) - dmu)/T) * (-1./T);
		logPrintf("Density of states at Fermi level = %le /eV-cell\n", dos0*(eV*detR));
		scaledDe = De / std::pow(dos0,3);
		
		//Perturb by photon-induced carrier density:
		//--- read carrier distributions from plasmonDecay:
		Histogram2D distribDirect("carrierDistribDirect.dat", 1./eV, 1./eV, 1.);
		Histogram2D distribPhonon("carrierDistribPhonon.dat", 1./eV, 1./eV, 1.);
		if(Eplasmon < distribDirect.omegaMin || Eplasmon > distribDirect.omegaMin + (distribDirect.nomega-1)*distribDirect.domega)
			die("Plasmon energy is out of the range available in carrierDistribDirect.dat")
		if(Eplasmon < distribPhonon.omegaMin || Eplasmon > distribPhonon.omegaMin + (distribPhonon.nomega-1)*distribPhonon.domega)
			die("Plasmon energy is out of the range available in carrierDistribPhonon.dat")
		//--- interpolate to required photon energy and carrier energy grid:
		dfPert.resize(nE);
		double Upert = 0.;
		double dZ = 0.;
		const double dnCut = 1e-5/Eplasmon; //number change threshold for active window
		ieMin = std::max(0, int(floor((-Eplasmon-10*T-dos.xMin)/dE)));
		ieMax = std::min(nE, int(ceil((Eplasmon+10*T-dos.xMin)/dE)));
		for(int ie=0; ie<nE; ie++)
		{	const double& Ei = Egrid(ie);
			double dni = 0.; //induced carrier number change at given energy
			for(int jE = -10; jE<=+10; jE++)
			{	double w = exp(-(jE*jE)/18.) / (sqrt(2*M_PI)*3); //gauss smoothing kernel with width 3*dE
				dni += w * (distribDirect.interp1(Ei+jE*dE, Eplasmon) + distribPhonon.interp1(Ei+jE*dE, Eplasmon));
			}
			//Smoothly truncate any spurious contributions beyond photon energy:
			if(fabs(Ei) > Eplasmon)
				dni *= exp(-0.5*std::pow((fabs(Ei) - Eplasmon)/(3*dE), 2));
			Upert += dni * Ei * dE; //calculate energy of perturbation
			if (Ei < minEcut || Ei > maxEcut)
			{	double dniInjected = dni * pInject;
				dZ -= dniInjected * dE; //count electrons/holes removed
				dni -= dniInjected;
			}
			//Include perturbation and update active range:
			if(fabs(dni) > dnCut)
			{	dfPert[ie] = dni / std::max(dos.yGrid[0][ie], 1e-3*dos0); //divide by DOS to get the effective filling change (regularize to avoid Infs)
				ieMin = std::min(ieMin, ie);
				ieMax = std::max(ieMax, ie);
			}
		}
		smooth(dfPert, nSmooth);
		dfPert *= Uabs / Upert; //normalize to match absorbed laser energy per unit volume
		dZ *= detR * Uabs / Upert; //correspondingly normalize (but per unit cell)
		logPrintf("Change in electrons/cell: %lg\n", dZ);

		//Electron-phonon coupling:
		Interp1 hIntInterp; hIntInterp.init("hEph.dat", eV, eV/pow(Angstrom,3));
		//--- interpolate to all the interval midpoints of energy grid:
		hInt.resize(nE-1);
		for(int ie=0; ie<nE-1; ie++)
			hInt[ie] = hIntInterp(Egrid(ie)+0.5*dE);
		smooth(hInt, nSmooth);
		
		//Divide active energy grid for e-e scattering calculation:
		//--- make length commensurate with eeStride
		mpiWorld->bcast(ieMin);
		mpiWorld->bcast(ieMax);
		nEcoarse = ceildiv(ieMax-ieMin+2, eeStride) - 1;
		ieMax = ieMin-2 + (nEcoarse+1)*eeStride;
		if(ieMax >= nE)
		{	ieMax -= eeStride;
			nEcoarse--;
		}
		assert(nEcoarse > 0);
		logPrintf("Active energy grid: [%d,%d) of total %d points\n", ieMin, ieMax, nE);
		//--- initialize coarse grid
		dEcoarse = dE * eeStride;
		TaskDivision(nEcoarse, mpiWorld).myRange(ieStart, ieStop);
		logPrintf("Coarse grid: %d points with [%d,%d) on current process.\n", nEcoarse, ieStart, ieStop);
		//--- initialize kernel:
		kernel.resize(2*eeStride-1);
		for(int i=0; i<int(kernel.size()); i++)
			kernel[i] = 1. - std::abs(i+1-eeStride)/eeStride;
		//--- coarse grain Mee and gM:
		Mc.resize(nEcoarse); gather(Mee.data(), Mc.data());
		gMc.resize(nEcoarse); gather(gM.data(), gMc.data());
		
		if(ip.dryRun)
		{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
			finalizeSystem();
			exit(0);
		}
		logPrintf("\n");
	}
	
	//Bisect for chemical potential:
	double get_dmu(double T)
	{	double dmuMin = dos.xGrid.front() - 10*T;
		double dmuMax = dos.xGrid.back() + 10*T;
		double dmu = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*T;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(int ie=0; ie<nE; ie++)
			{	double& fi = f0[ie];
				fi = fermi((Egrid(ie) - dmu)/T);
				nElectrons += dE * dos.yGrid[0][ie] * fi * detR;
			}
			((nElectrons>Z) ? dmuMax : dmuMin) = dmu;
			dmu = 0.5*(dmuMin + dmuMax);
		}
		return dmu;
	}
	
	//Get thermal energy at a given chemical potential:
	double get_Uthermal(double T)
	{	get_dmu(T); //this sets the Fermi distribution in f0
		double U = 0.;
		for(int ie=0; ie<nE; ie++)
			U += dE * dos.yGrid[0][ie] * f0[ie] * Egrid(ie);
		return U;
	}
	
	//Compute df/dt given f (nonlinear e-e and e-Ph collision integrals):
	diagMatrix compute(double t, const diagMatrix& f)
	{	diagMatrix fdot(nE+1); //last entry is TlDot
		double& TlDot = fdot.back();
		const double Tl = f.back();
		
		//e-e collisions:
		//--- gather f to coarse grid:
		diagMatrix fc(nEcoarse);
		gather(f.data(), fc.data());
		//--- compute fdot:
		diagMatrix fcDot(nEcoarse);
		for(int i=ieStart; i<ieStop; i++)
		{	double rateSum = 0.;
			for(int i1=0; i1<nEcoarse; i1++)
			{	double inOcc = fc[i]*fc[i1];
				double inUnocc = (1.-fc[i])*(1.-fc[i1]);
				//i2 range set by both i2 and i3 in [0,nEcoarse)
				int i2min = std::max(i+i1+1-nEcoarse, 0);
				int i2max = std::min(i+i1+1, nEcoarse);
				for(int i2=i2min; i2<i2max; i2++)
				{	int i3 = i+i1-i2; //energy conservation
					double outOcc = fc[i2]*fc[i3];
					double outUnocc = (1.-fc[i2])*(1.-fc[i3]);
					rateSum += (inUnocc*outOcc - inOcc*outUnocc) * gMc[i1]*gMc[i2]*gMc[i3];
				}
			}
			fcDot[i] = scatterFactor * (2*scaledDe) * (dEcoarse*dEcoarse) * rateSum * Mc[i];
		}
		mpiWorld->allReduceData(fcDot, MPIUtil::ReduceSum);
		//--- scatter fdot to fine grid:
		scatter(fcDot.data(), fdot.data());
		
		//e-ph collisions:
		double ElDot = 0.; //rate of energy transfer to lattice
		const double* g = dos.yGrid[0].data(); //DOS data pointer
		for(int i=0; i<nE-1; i++)
		{	if(std::min(g[i],g[i+1]) < 1e-3*dos0) continue; //ignore intervals with no electrons to avoid division by zero below
			double fPrime = (f[i+1]-f[i])/dE;
			double fMean = 0.5*(f[i+1]+f[i]);
			double ElDot_i = scatterFactor * (2*M_PI*dE) * hInt[i] * (fMean*(1.-fMean) + fPrime*Tl); //rate of energy transfer to lattice from this interval
			ElDot += ElDot_i;
			fdot[i] += ElDot_i / (dE*dE*g[i]);
			fdot[i+1] -= ElDot_i / (dE*dE*g[i+1]);
		}
		TlDot = fixedTl ? 0.0 : ElDot / Cl(Tl);
		
		//Pump evolution:
		if(pumpCW or pumpFWHM)
		{	double prefactor;
			if(pumpCW)
				prefactor = 1.0; //CW
			else
			{	double sigma = pumpFWHM / 2.355;
				prefactor = exp(-0.5*std::pow(t/sigma, 2)) / (sigma * sqrt(2*M_PI));
			}
			for(int i=0; i<nE; i++)
				fdot[i] += prefactor * dfPert[i];
		}
		
		mpiWorld->bcastData(fdot); //Ensure consistency on all processes
		return fdot;
	}
	
	//Per time-step reporting:
	void report(double t, const diagMatrix& f) const
	{	const double Eunits = Joule/pow(meter,3);
		double dEe = Ee(f) - Ee0;
		double dEl = El(f.back()) - El0;
		logPrintf("Integrate:  t[fs]: %6.1lf   Ee[J/m^3]: %14.8le   El[J/m^3]: %14.8le   Etot[J/m^3]: %14.8le   Tl[K]: %7.2lf\n",
			t/fs, dEe/Eunits, dEl/Eunits, (dEe+dEl)/Eunits, f.back()/Kelvin);
		logFlush();
		//Store time and corresponding results:
		ePhRelax& e = *((ePhRelax*)this);
		e.tArr.push_back(t);
		e.fArr.push_back(f);
	}
	
	//Evaluate e-e linewidth correction:
	diagMatrix linewidthCorrection(const diagMatrix& f) const
	{	//Linewidth correction within jellium model depends only on energy in electronic system:
		double result = 0.5*De*std::pow(M_PI*T, 2);
		for(int i=ieMin; i<ieMax; i++)
			result += (3.*De*dE) * Egrid(i) * (f[i] - f0[i]);
		return diagMatrix(nE, result);
	}
	
	//Calculate lattice specific heat
	inline double Cl(double Tl) const
	{	assert(dosPh.xMin==0.);
		const double& domegaPh = dosPh.dx;
		double result = 0.;
		for(size_t ie=1; ie<dosPh.xGrid.size(); ie++) //omit zero energy phonons to avoid 0/0 error
		{	double omegaPh = ie*domegaPh;
			double g = bose(omegaPh/Tl);
			double g_Tl = g*(g+1)*omegaPh/(Tl*Tl); //dg/dTl
			result += domegaPh * omegaPh * g_Tl  * dosPh.yGrid[0][ie];
		}
		return result;
	}
	
	//Calculate lattice energy density:
	inline double El(double Tl) const
	{	assert(dosPh.xMin==0.);
		const double& domegaPh = dosPh.dx;
		double result = 0.;
		for(size_t ie=1; ie<dosPh.xGrid.size(); ie++) //omit zero energy phonons to avoid 0/0 error
		{	double omegaPh = ie*domegaPh;
			double g = bose(omegaPh/Tl);
			result += domegaPh * omegaPh * g  * dosPh.yGrid[0][ie];
		}
		return result;
	}
	
	//Calculate electronic energy density:
	inline double Ee(const diagMatrix& f) const
	{	double result = 0.;
		for(int ie=0; ie<nE; ie++)
			result += dE * Egrid(ie) * f[ie]  * dos.yGrid[0][ie];
		return result;
	}
};

int main(int argc, char** argv)
{	ePhRelax e(argc, argv);
	
	//Solve time dependence:
	StopWatch watchSolve("Solve"); watchSolve.start();
	e.Ee0 = e.Ee(e.f0);
	e.El0 = e.El(e.T);
	diagMatrix f = e.f0; f.push_back(e.T); //Initial distribution
	double t;
	if(e.pumpCW)
	{	t = 0.; //CW pump applied during integration (and no extra time needed at start)
	}
	else if(e.pumpFWHM == 0.)
	{	e.report(-e.dt, f);
		//Apply pump instanteneously:
		t = 0.;
		f = e.f0 + e.dfPert; f.push_back(e.T);
	}
	else
	{	t = -e.dt * ceil(2.*e.pumpFWHM/e.dt); //start earlier to accomodate Gaussian pump (applied during integration)
	}
	e.integrateAdaptive(f, t, e.tMax, 1e-4, e.dt);
	watchSolve.stop();
	
	//Calculate carrier linewidths and effective temperature:
	logPrintf("Calculating linewidths ... "); logFlush();
	StopWatch watchLinewidths("Linewidths"); watchLinewidths.start();
	std::vector<diagMatrix> lwDelta;
	for(diagMatrix& f: e.fArr)
		lwDelta.push_back(e.linewidthCorrection(f));
	watchLinewidths.stop();
	logPrintf("done.\n");

	//File outputs:
	if(mpiWorld->isHead())
	{	std::ofstream ofs;
		
		//Lattice temperature:
		ofs.open((e.runName+".Tl").c_str());
		ofs.precision(10);
		ofs << "#t[fs] Tl[K]\n";
		for(int it=0; it<int(e.tArr.size()); it++)
			ofs << e.tArr[it]/fs << '\t' << e.fArr[it].back()/Kelvin << '\n';
		ofs.close();
		
		//Pump perturbation distribution [dimensionless]
		ofs.open((e.runName+".dfPert").c_str());
		ofs.precision(10);
		ofs << "#E[ev]\\tdfPert\n";
		for(int ie=0; ie<e.nE; ie++)
			ofs << e.Egrid(ie)/eV << '\t' << e.dfPert[ie] << '\n';
		ofs.close();
		
		//Distributions [dimensionless]
		ofs.open((e.runName+".f").c_str());
		ofs.precision(10);
		//--- Header
		ofs << "#E[ev]\\t[fs]";
		for(int it=0; it<int(e.tArr.size()); it++)
			ofs << '\t' << e.tArr[it]/fs;
		ofs << '\n';
		//--- Data
		for(int ie=0; ie<e.nE; ie++)
		{	ofs << e.Egrid(ie)/eV;
			for(size_t it=0; it<e.tArr.size(); it++)
				ofs << '\t' << e.fArr[it][ie];
			ofs << '\n';
		}
		ofs.close();
		
		//Linewidth corrections [eV]
		ofs.open((e.runName+".lwDelta").c_str());
		ofs.precision(10);
		//--- Header
		ofs << "#E[ev]\\t[fs]";
		for(int it=0; it<int(e.tArr.size()); it++)
			ofs << '\t' << e.tArr[it]/fs;
		ofs << '\n';
		//--- Data
		for(int ie=0; ie<e.nE; ie++)
		{	ofs << e.Egrid(ie)/eV;
			for(size_t it=0; it<e.tArr.size(); it++)
				ofs << '\t' << lwDelta[it][ie]/eV;
			ofs << '\n';
		}
		ofs.close();
	}
	
	FeynWann::finalize();
};
