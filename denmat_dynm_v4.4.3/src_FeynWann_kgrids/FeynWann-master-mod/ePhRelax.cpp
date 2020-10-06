#include <core/Units.h>
#include "FeynWann.h"
#include "InputMap.h"
#include "Interp1.h"
#include "Histogram.h"
#include <core/Util.h>
#include <core/Operators.h>
#include <core/matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <fstream>

struct ePhRelax
{
	Interp1 dos, dosPh;
	diagMatrix f0, fPert; //Initial Fermi and photon-perturbed distributions
	double dt, tMax; //time step for output and max time
	double Z, detR; //electrons and volume per unit cell
	double T, dos0; //initial temperature and density of states at the Fermi level
	double minEcut, maxEcut; //E cutoff in eV below/above which holes/electron distribution is zero
	double pInject; //probability that a carrier gets injected to substrate
	double De, scaledDe; //De, and De scaled by g(eF)**-3
	diagMatrix hInt; //energy resolved electron-phonon coupling
	string runName;
	
	//Energy grid:
	int nE; double Emin, dE;
	inline double Egrid(int i) const { return dos.xGrid[i]; }
	int ieMin, ieMax; //min and max active energy grid indices (that evolve with time)
	int ieStart, ieStop; //min and max energy grid indices to deal with on current MPI process
	
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
		minEcut = inputMap.get("minEcut")*eV; // energy cutoff below which hole distribution is zero
		maxEcut = inputMap.get("maxEcut")*eV; //energy cutoff below which hole distribution is zero
		const double Uabs = inputMap.get("Uabs") * Joule/std::pow(meter,3); //absorbed laser energy per unit volume in Joule/meter^3
		const double Eplasmon = inputMap.get("Eplasmon") * eV; //incident photon energy in eV
		De = inputMap.get("De") / eV; //quadratic e-e lifetime coefficient in eV^-1
		runName = inputMap.getString("runName"); //prefix to use for output files
		const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
		detR = fabs(det(R));
		
		logPrintf("\nInputs after conversion to atomic units:\n");
		logPrintf("dt = %lg\n", dt);
		logPrintf("tMax = %lg\n", tMax);
		logPrintf("Z = %lg\n", Z);
		logPrintf("T = %lg\n", T);
		logPrintf("E range in carrier distrib = (%lg, %lg)\n", minEcut, maxEcut);
		logPrintf("Probability a carrier gets injected: %lg\n", pInject);
		logPrintf("Uabs = %lg\n", Uabs);
		logPrintf("Eplasmon = %lg\n", Eplasmon);
		logPrintf("De = %lg\n", De);
		logPrintf("runName = %s\n", runName.c_str());
		logPrintf("R:\n"); R.print(globalLog, " %lg ");
		logPrintf("detR = %lg\n", detR);
		
		//Read electron and phonon DOS (and convert to atomic units and per-unit volume):
		dos.init("eDOS.dat", eV, 1./(detR*eV));
		dosPh.init("phDOS.dat", eV, 1./(detR*eV));
		nE = dos.xGrid.size();
		dE = dos.dx;
		f0.resize(nE);
		
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
		//--- interpolate to required photon energy and carrier eenergy grid:
		fPert.resize(nE);
		double Upert = 0.;
		double dZ = 0.;
		for(int ie=0; ie<nE; ie++)
		{	const double& Ei = Egrid(ie);
			double dni = distribDirect.interp1(Ei, Eplasmon) + distribPhonon.interp1(Ei, Eplasmon); //induced carrier number change at given energy
			Upert += dni * Ei * dE; //calculate energy of perturbation
			if (Ei < minEcut || Ei > maxEcut)
			{	double dniInjected = dni * pInject;
				dZ -= dniInjected * dE; //count electrons/holes removed
				dni -= dniInjected;
			}
			fPert[ie] = dni / std::max(dos.yGrid[0][ie], 1e-3*dos0); //divide by DOS to get the effective filling change (regularize to avoid Infs)
		}
		fPert *= Uabs / Upert; //normalize to match absorbed laser energy per unit volume
		dZ *= detR * Uabs / Upert; //correspondingly normalize (but per unit cell)
		fPert += f0; //add initial Fermi distribution
		logPrintf("Change in electrons/cell: %lg\n", dZ);

		//Electron-phonon coupling:
		Interp1 hIntInterp; hIntInterp.init("hEph.dat", eV, eV/pow(Angstrom,3));
		//--- interpolate to all the interval midpoints of energy grid:
		hInt.resize(nE-1);
		for(int ie=0; ie<nE-1; ie++)
			hInt[ie] = hIntInterp(Egrid(ie)+0.5*dE);
		
		//Determine active energy grid:
		ieMin = std::max(0, int(floor((-Eplasmon-10*T-dos.xMin)/dE)));
		ieMax = std::min(nE, int(ceil((Eplasmon+10*T-dos.xMin)/dE)));
		int neActive = ieMax - ieMin;
		TaskDivision(neActive, mpiWorld).myRange(ieStart, ieStop);
		ieStart += ieMin;
		ieStop += ieMin;
		logPrintf("Active energy grid: [%d,%d) of total %d points, with [%d,%d) on current process.\n", ieMin, ieMax, nE, ieStart, ieStop);
		
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
	
	//Evaluate e-e and e-Ph collision integrals (nonlinear):
	diagMatrix fdot(const diagMatrix& f) const
	{	diagMatrix results(nE+1); //last entry is TlDot
		double& TlDot = results.back();
		const double Tl = f.back();
		const double* g = dos.yGrid[0].data(); //DOS data pointer
		//e-e collisions:
		for(int i=ieStart; i<ieStop; i++)
		{	double rateSum = 0.;
			for(int i1=ieMin; i1<ieMax; i1++)
			{	double inOcc = f[i]*f[i1];
				double inUnocc = (1.-f[i])*(1.-f[i1]);
				//i2 range set by both i2 and i3 in [ieMin,ieMax)
				int i2min = std::max(i+i1+1-ieMax, ieMin);
				int i2max = std::min(i+i1+1-ieMin, ieMax);
				for(int i2=i2min; i2<i2max; i2++)
				{	int i3 = i+i1-i2; //energy conservation
					double outOcc = f[i2]*f[i3];
					double outUnocc = (1.-f[i2])*(1.-f[i3]);
					rateSum += (inUnocc*outOcc - inOcc*outUnocc) * g[i1]*g[i2]*g[i3];
				}
			}
			results[i] = (2*scaledDe) * (dE*dE) * rateSum;
		}
		mpiWorld->allReduceData(results, MPIUtil::ReduceSum, true);
		//e-ph collisions:
		double ElDot = 0.; //rate of energy transfer to lattice
		for(int i=0; i<nE-1; i++)
		{	if(std::min(g[i],g[i+1]) < 1e-3*dos0) continue; //ignore intervals with no electrons to avoid division by zero below
			double fPrime = (f[i+1]-f[i])/dE;
			double fMean = 0.5*(f[i+1]+f[i]);
			double ElDot_i = (2*M_PI*dE) * hInt[i] * (fMean*(1.-fMean) + fPrime*Tl); //rate of energy transfer to lattice from this interval
			ElDot += ElDot_i;
			results[i] += ElDot_i / (dE*dE*g[i]);
			results[i+1] -= ElDot_i / (dE*dE*g[i+1]);
		}
		TlDot = ElDot / Cl(Tl);
		return results;
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

//Wrapper function for GSL integrator:
int fdot_wrapper(double t, const double* f, double* fdot, void* params)
{	const ePhRelax& e = *((ePhRelax*)params);
	diagMatrix fMat; fMat.assign(f, f+e.nE+1); //copy input to diagMatrix
	diagMatrix fdotMat = e.fdot(fMat); //calculate result in diagMatrix form
	mpiWorld->bcastData(fdotMat); //make sure results are consistent across processes to numerical precision
	std::copy(fdotMat.begin(), fdotMat.end(), fdot); //copy output to pointer
	return GSL_SUCCESS;
}

int main(int argc, char** argv)
{	ePhRelax e(argc, argv);
	
	//Solve time dependence:
	std::vector<diagMatrix> fArr;
	StopWatch watchSolve("Solve"); watchSolve.start();
	gsl_odeiv2_system odeSystem = {fdot_wrapper, NULL, size_t(e.nE+1), &e };
	gsl_odeiv2_driver* odeDriver = gsl_odeiv2_driver_alloc_y_new(&odeSystem, gsl_odeiv2_step_msadams, 1e-6, 1e-6, 0.0);
	double Ee0 = e.Ee(e.f0), El0 = e.El(e.T);
	//--- t = -dt: just before absorption
	double t = -e.dt;
	diagMatrix f = e.f0; f.push_back(e.T);
	fArr.push_back(f);
	//--- t = 0: just after absorption
	t = 0;
	f = e.fPert; f.push_back(e.T);
	fArr.push_back(f);
	logPrintf("\nSolving boltzmann eqn:\n");
	logPrintf("%5s  %19s  %19s  %19s  %7s  %s\n", "t[fs]",  "Ee[J/m^3]", "El[J/m^3]", "(El+Ee)[J/m^3]", "Tl[K]", "Progress");
	logFlush();
	
	while(t < e.tMax-1e-3*e.dt)
	{	int status = gsl_odeiv2_driver_apply(odeDriver, &t, t+e.dt, f.data());
		if(status != GSL_SUCCESS) die("Error %d in ODE propagation", status)
		fArr.push_back(f);
		
		//Print progress:
		const double Eunits = Joule/pow(meter,3);
		double dEe = e.Ee(f)-Ee0, dEl = e.El(f.back())-El0;
		logPrintf("%5g  %19.13le  %19.13le  %19.13le  %7.2lf  %.1f%%\n", t/fs, dEe/Eunits, dEl/Eunits, (dEe+dEl)/Eunits, f.back()/Kelvin, 100.*t/e.tMax);
		logFlush();
	}
	logPrintf("done.\n"); logFlush();
	gsl_odeiv2_driver_free(odeDriver);
	watchSolve.stop();
	
	//Calculate carrier linewidths and effective temperature:
	logPrintf("\nCalculating linewidths ... "); logFlush();
	StopWatch watchLinewidths("Linewidths"); watchLinewidths.start();
	std::vector<diagMatrix> lwDelta;
	for(diagMatrix& f: fArr)
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
		for(int it=0; it<int(fArr.size()); it++)
			ofs << ((it-1)*e.dt)/fs << '\t' << fArr[it].back()/Kelvin << '\n';
		ofs.close();
		
		//Distributions [dimensionless]
		ofs.open((e.runName+".f").c_str());
		ofs.precision(10);
		//--- Header
		ofs << "#E[ev]\\t[fs]";
		for(int it=0; it<int(fArr.size()); it++)
			ofs << '\t' << ((it-1)*e.dt)/fs;
		ofs << '\n';
		//--- Data
		for(int ie=0; ie<e.nE; ie++)
		{	ofs << e.Egrid(ie)/eV;
			for(size_t it=0; it<fArr.size(); it++)
				ofs << '\t' << fArr[it][ie];
			ofs << '\n';
		}
		ofs.close();
		
		//Linewidth corrections [eV]
		ofs.open((e.runName+".lwDelta").c_str());
		ofs.precision(10);
		//--- Header
		ofs << "#E[ev]\\t[fs]";
		for(int it=0; it<int(fArr.size()); it++)
			ofs << '\t' << ((it-1)*e.dt)/fs;
		ofs << '\n';
		//--- Data
		for(int ie=0; ie<e.nE; ie++)
		{	ofs << e.Egrid(ie)/eV;
			for(size_t it=0; it<fArr.size(); it++)
				ofs << '\t' << lwDelta[it][ie]/eV;
			ofs << '\n';
		}
		ofs.close();
	}
	
	finalizeSystem();
};
