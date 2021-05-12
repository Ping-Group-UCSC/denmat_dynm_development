#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include <core/Units.h>
#include <commands/command.h>
#include "InputMap.h"
#include "Histogram.h"
#include "FeynWann.h"


//Get energy range from an eLoop call:
struct EnergyRange
{	double Emin;
	double Emax;
	
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	EnergyRange& er = *((EnergyRange*)params);
		er.Emin = std::min(er.Emin, state.E.front()); //E is in ascending order
		er.Emax = std::max(er.Emax, state.E.back()); //E is in ascending order
	}
};

//Collect ImEps contibutions using FeynWann callbacks:
struct CollectHePh
{	double dE;
	Histogram hEph, dos;
	double prefac, prefacDOS;
	bool useLinewidths;
	
	CollectHePh(double Emin, double dE, double Emax, bool useLinewidths)
	: dE(dE), hEph(Emin, dE, Emax), dos(Emin, dE, Emax), useLinewidths(useLinewidths)
	{	logPrintf("Initialized energy grid: %lg to %lg eV with %d points.\n", hEph.Emin/eV, hEph.Emax()/eV, hEph.nE);
	}
	
	void calcLinewidth(const FeynWann::StateE& state, diagMatrix& ImE)
	{	assert(useLinewidths);
		int nBands = state.E.nRows();
		ImE = state.ImSigma_ee; //e-e part
		for(int b=0; b<nBands; b++)
			ImE[b] += state.ImSigma_ePh(b, state.E[b]<0. ? 1. : 0.); //e-ph part
	}
	
    void collect(const FeynWann::MatrixEph& mat)
	{	int nBands = mat.e1->E.nRows();
		//Get energies and linewidths:
		const diagMatrix& E1 = mat.e1->E;
		const diagMatrix& E2 = mat.e2->E;
		diagMatrix ImE1, ImE2;
		if(useLinewidths)
		{	calcLinewidth(*mat.e1, ImE1);
			calcLinewidth(*mat.e2, ImE2);
		}
		const diagMatrix& omegaPh = mat.ph->omega;
		int nModes = omegaPh.nRows();
		//Collect
		for(int v=0; v<nBands; v++)
		{	for(int c=0; c<nBands; c++)
			{	for(int alpha=0; alpha<nModes; alpha++)
				{	double gePhSq = mat.M[alpha](c,v).norm();
					double breadth = useLinewidths ? (ImE1[v] + ImE2[c]) : dE;
					double delta = 1./(M_PI*breadth*(1. + std::pow((E1[v]-E2[c] + omegaPh[alpha])/breadth,2)));
					hEph.addEvent(0.5*(E1[v]+E2[c]), prefac * delta * omegaPh[alpha] * gePhSq);
				}
			}
			dos.addEvent(E1[v], 0.5*prefacDOS);
			dos.addEvent(E2[v], 0.5*prefacDOS);
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mat, void* params)
	{	((CollectHePh*)params)->collect(mat);
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Energy-resolved electron-phonon coupling strength");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double Tmin = inputMap.get("Tmin") * Kelvin; //electron temperature grid start
	const double Tmax = inputMap.get("Tmax") * Kelvin; //electron temperature grid stop
	const double Tstep = inputMap.get("Tstep") * Kelvin; //electron temperature grid spacing
	bool useLinewidths = false;
	boolMap.getEnum(inputMap.getString("useLinewidths").c_str(), useLinewidths); //whether to use linewidths for broadening
	FeynWannParams fwp(&inputMap);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("dE = %lg\n", dE);
	logPrintf("Tmin = %lg\n", Tmin);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("Tstep = %lg\n", Tstep);
	logPrintf("useLinewidths = %s\n", boolMap.getString(useLinewidths));
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.needPhonons = true;
	fwp.needLinewidth_ee = useLinewidths;
	fwp.needLinewidth_ePh = useLinewidths;
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKeff = nOffsets * fw->ePhCountPerOffset();
	logPrintf("Effectively sampled nKpairs: %lu\n", nKeff);
	if(mpiWorld->isHead()) logPrintf("%d phonon q-mesh offset pairs parallelized over %d process groups.\n", nOffsets, mpiGroupHead->nProcesses());

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize temperature grid:
	std::vector<double> Tarr(int(ceil((Tmax-Tmin)/Tstep)));
	for(size_t iT=0; iT<Tarr.size(); iT++)
		Tarr[iT] = Tmin + Tstep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", Tarr.front()/Kelvin, Tarr.back()/Kelvin, Tarr.size());

	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Initialize energy grid:
	EnergyRange er = { DBL_MAX, -DBL_MAX };
	for(vector3<> qOff: fw->qOffset) fw->eLoop(qOff, EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	er.Emin = dE * (floor(er.Emin/dE) - 10); //add some margin and ensure grid contains 0
	er.Emax = dE * (ceil(er.Emax/dE) + 10);
	
	//Collect e-ph coupling resolved by energy:
	CollectHePh ch(er.Emin, dE, er.Emax, useLinewidths);
	ch.prefac = fw->spinWeight / (nKeff*fabs(det(fw->R)));
	ch.prefacDOS = fw->spinWeight * (1./nKeff);
	for(int iSpin=0; iSpin<fw->nSpins; iSpin++)
	{	//Update FeynWann for spin channel if necessary:
		if(iSpin>0)
		{	fw = 0; //free memory from previous spin
			fwp.iSpin = iSpin;
			fw = std::make_shared<FeynWann>(fwp);
		}
		logPrintf("\nCollecting hEph: "); logFlush();
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			vector3<> k01 = fw->randomVector(mpiGroup); //must be constant across group
			vector3<> k02 = fw->randomVector(mpiGroup); //must be constant across group
			fw->ePhLoop(k01, k02, CollectHePh::ePhProcess, &ch);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	ch.dos.allReduce(MPIUtil::ReduceSum);
	ch.hEph.allReduce(MPIUtil::ReduceSum);
	ch.hEph.print("hEph.dat", 1./eV, 1./(eV/pow(Angstrom,3)));
	
	//Calculate GePh from hEph for various Te:
	logPrintf("\nCalculating GePh: "); logFlush();
	diagMatrix GePh(Tarr.size(), 0.);
	//--- check enough bands to contain Z:
	double Zmax = 0.;
	for(const double& g: ch.dos.out)
		Zmax += dE * g;
	if(Zmax < fw->nElectrons)
		die("Current DOS can only support %lg electrons > %lg electrons specified.\n", Zmax, fw->nElectrons);
	int iTstart, iTstop; TaskDivision(Tarr.size(), mpiWorld).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double T = Tarr[iT], invT = 1./T;
		//Bisect for chemical potential:
		double dmuMin = ch.dos.Emin - 10*T;
		double dmuMax = ch.dos.Emax() + 10*T;
		double dmu = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*T;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(int ie=0; ie<ch.dos.nE; ie++)
			{	double Ei = ch.dos.Emin + ie*dE;
				double fi = fermi(invT*(Ei - dmu));
				nElectrons += dE * ch.dos.out[ie] * fi;
			}
			((nElectrons>fw->nElectrons) ? dmuMax : dmuMin) = dmu;
			dmu = 0.5*(dmuMin + dmuMax);
		}
		//Calculate e-ph coupling for each Te:
		double& Gcur = GePh[iT];
		Gcur = 0.;
		for(int ie=0; ie<ch.dos.nE; ie++)
		{	double Ei = ch.dos.Emin + ie*dE;
			double x = invT*(Ei-dmu);
			double dfdE = (-invT) * fermiPrime(x);
			Gcur += dE * ch.hEph.out[ie] * dfdE;
		}
		Gcur *= (2*M_PI);
	}
	mpiWorld->allReduceData(GePh, MPIUtil::ReduceSum);
	//--- write to file
	if(mpiWorld->isHead())
	{	const double GePhSI = Joule/(Kelvin*pow(meter,3)*sec);
		ofstream ofs("GePh.dat");
		ofs << "#T[K] GePh[W/m^3K]\n";
		for(size_t iT=0; iT<Tarr.size(); iT++)
			ofs << Tarr[iT]/Kelvin << '\t'
				<< GePh[iT]/GePhSI << '\n';
	}
	logPrintf("done.\n"); logFlush();
	
	fw = 0;
	FeynWann::finalize();
}
