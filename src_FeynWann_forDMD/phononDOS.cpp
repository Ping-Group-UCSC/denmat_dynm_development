/*-------------------------------------------------------------------
Copyright 2018 Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#include "FeynWann.h"
#include "InputMap.h"
#include "Histogram.h"
#include <core/Units.h>
#include <core/Random.h>
#include "lindbladInit_for-DMD-4.5.6/help.h"

void findMaxOmega(const FeynWann::StatePh& state, void* params)
{	double& omegaMax = *((double*)params);
	omegaMax = std::max(omegaMax, state.omega.back()); //omega is in ascending order
}

struct CollectDOS{
	Histogram* dos;
	double weight;
	FeynWann fw;
	bool atomProj;
	std::vector<Histogram> dosAtom;
	diagMatrix basisLength;
	double qmax;

	CollectDOS(FeynWann& fw, bool atomProj, double domega, double omegaMax)
		: fw(fw), atomProj(atomProj), dosAtom(fw.nAtoms, Histogram(0, domega, omegaMax)), basisLength(fw.nModes)
	{
		if (atomProj){
			for (int m = 0; m < fw.nModes; m++)
				basisLength[m] = fw.phononBasis[m].length();
		}
	}
	
	void process(const FeynWann::StatePh& state){
		for (const double& omega : state.omega)
			dos->addEvent(omega, weight);

		if (!atomProj) return;
		double qlength = sqrt(fw.GGT.metric_length_squared(wrap(state.q)));
		if (qlength > qmax) return;
		matrix ionDisp = basisLength * state.U;
		complex* ionDispData = ionDisp.data();
		for (int m = 0; m < fw.nModes; m++){
			double omega = state.omega[m];

			std::vector<double> ionDispMag(fw.nAtoms, 0);
			double sum = 0;
			for (int ia = 0; ia < fw.nAtoms; ia++){
				for (int d = 0; d < 3; d++){
					ionDispMag[ia] += pow((*ionDispData).real(), 2.);
					ionDispData++;
				}
				ionDispMag[ia] = sqrt(ionDispMag[ia]);
				sum = sum + ionDispMag[ia];
			}

			for (int ia = 0; ia < fw.nAtoms; ia++)
				dosAtom[ia].addEvent(omega, weight * ionDispMag[ia] / sum);
		}
	}
	static void phProcess(const FeynWann::StatePh& state, void* params){
		((CollectDOS*)params)->process(state);
	}
};

int main(int argc, char** argv)
{	InitParams ip = FeynWann::initialize(argc, argv, "Phonon DOS and heat capacity");
	
	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	size_t nOffsets = inputMap.get("nOffsets");
	const double domega = inputMap.get("domega") * eV; //phonon energy resolution (should be much smaller than TD)
	const double Tmin = inputMap.get("Tmin") * Kelvin; //lattice temperature grid start
	const double Tmax = inputMap.get("Tmax") * Kelvin; //lattice temperature grid stop
	const double Tstep = inputMap.get("Tstep") * Kelvin; //lattice temperature grid spacing
	const double vL = inputMap.get("vL", 0.) * meter/sec; //longitudinal speed of sound (optional for Debye model)
	const double vT = inputMap.get("vT", vL) * meter/sec; //transverse speed of sound (assumed x2, optional for Debye model)
	const bool atomProj = inputMap.get("atomProj", 0);
	const double qmax = inputMap.get("qmax", 10);
	FeynWannParams fwp(&inputMap);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %lu\n", nOffsets);
	logPrintf("domega = %lg\n", domega);
	logPrintf("Tmin = %lg\n", Tmin);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("Tstep = %lg\n", Tstep);
	if(vL) logPrintf("vL = %lg\n", vL);
	if(vT) logPrintf("vT = %lg\n", vT);
	logPrintf("atomProj = %d\n", atomProj);
	logPrintf("qmax = %lg\n", qmax);
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.needPhonons = true;
	FeynWann fw(fwp);
	size_t nKpts = nOffsets * fw.phCountPerOffset();
	logPrintf("Effectively sampled nKpts: %lu\n", nKpts);
	if(mpiWorld->isHead()) logPrintf("%lu phonon q-mesh offsets parallelized over %d process groups.\n", nOffsets, mpiGroupHead->nProcesses());

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Calculate Debye temperatures / energies (same in atomic units):
	const double kD  = std::pow(6*M_PI*M_PI/fw.Omega, 1./3);
	const double TdebyeL = vL * kD; if(vL) logPrintf("Longitudinal Debye energy: %3.0lf K (%.1lf meV)\n", TdebyeL/Kelvin, TdebyeL/(1e-3*eV));
	const double TdebyeT = vT * kD; if(vT) logPrintf("Transverse Debye energy:   %3.0lf K (%.1lf meV)\n", TdebyeT/Kelvin, TdebyeT/(1e-3*eV));

	//Initialize temperature grid:
	std::vector<double> Tarr(int(ceil((Tmax-Tmin)/Tstep)));
	for(size_t iT=0; iT<Tarr.size(); iT++)
		Tarr[iT] = Tmin + Tstep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", Tarr.front()/Kelvin, Tarr.back()/Kelvin, Tarr.size());
	
	//Initialize phonon energy grid:
	double omegaMax = std::max(TdebyeL, TdebyeT);
	fw.phLoop(vector3<>(), findMaxOmega, &omegaMax);
	mpiWorld->allReduce(omegaMax, MPIUtil::ReduceMax);
	omegaMax *= 1.25; //add some margin
	Histogram dos(0, domega, omegaMax); //phonon density of states

	logPrintf("Initialized phonon energy grid: 0 to %lg eV with %lu points.\n", (domega*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	if(fw.nModes!=3 && vL) logPrintf("WARNING: the Debye estimates are only valid if nModes = 3.\n");
	
	logPrintf("\nCollecting DOS: "); logFlush();
	CollectDOS cd(fw, atomProj, domega, omegaMax);
	cd.dos = &dos;
	cd.weight = (1./nKpts);
	cd.qmax = qmax;
	for(int o=0; o<noMine; o++)
	{	Random::seed(o+oStart); //to make results independent of MPI division
		//Process with a random offset:
		vector3<> q0 = fw.randomVector(mpiGroup); //must be constant across group
		fw.phLoop(q0, CollectDOS::phProcess, &cd);
		//Print progress:
		if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	dos.allReduce(MPIUtil::ReduceSum);
	dos.print("phDOS.dat", 1./eV, eV);
	if (atomProj){
		for (int ia = 0; ia < fw.nAtoms; ia++){
			cd.dosAtom[ia].allReduce(MPIUtil::ReduceSum);
			string fname = "phDOS_at" + int2str(ia) + ".dat";
			cd.dosAtom[ia].print(fname, 1. / eV, eV);
		}
	}
	
	//Calculate Cl at each temperature:
	diagMatrix Cl(Tarr.size(), 0.), ClDebye(Tarr.size(), 0.);
	int iTstart, iTstop; TaskDivision(Tarr.size(), mpiWorld).myRange(iTstart, iTstop);
	const double dosPrefacDebyeL = fw.Omega / (2*M_PI*M_PI * std::pow(vL,3)); 
	const double dosPrefacDebyeT = fw.Omega / (2*M_PI*M_PI * std::pow(vT,3)); 
	std::vector<double> dosDebyeArr(dos.out.size());
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double T = Tarr[iT], invT = 1./T;
		
		//Calculate lattice specific heat using DOS:
		double& ClCur = Cl[iT]; ClCur = 0.;
		double& ClDebyeCur = ClDebye[iT]; ClDebyeCur = 0.;
		for(size_t ie=1; ie<dos.out.size(); ie++) //omit zero energy phonons to avoid 0/0 error
		{	double omegaPh = ie*domega;
			double g = bose(invT*omegaPh);
			double g_T = g*(g+1)*omegaPh*invT*invT; //dg/dT
			ClCur += domega * omegaPh * g_T  * dos.out[ie];
			//Debye approximation:
			double dosDebyeL = dosPrefacDebyeL * (omegaPh<TdebyeL ? omegaPh*omegaPh : 0.);
			double dosDebyeT = dosPrefacDebyeT * (omegaPh<TdebyeT ? omegaPh*omegaPh : 0.); //per mode
			double dosDebye = dosDebyeL + 2*dosDebyeT; //2 transverse modes
			dosDebyeArr[ie] = dosDebye;
			ClDebyeCur += domega * omegaPh * g_T  * dosDebye;
		}
	}
	mpiWorld->allReduceData(Cl, MPIUtil::ReduceSum);
	mpiWorld->allReduceData(ClDebye, MPIUtil::ReduceSum);

	if(mpiWorld->isHead())
	{	const double Omega = fw.Omega;
		const double ClSI = Joule/(Kelvin*pow(meter,3));
		ofstream ofs("Cl.dat");
		ofs << "#T[K] Cl[J/m^3K] ClDebye[J/m^3K]\n";
		for(size_t iT=0; iT<Tarr.size(); iT++)
			ofs << Tarr[iT]/Kelvin << '\t'
				<< Cl[iT]/(Omega*ClSI) << '\t'
				<< ClDebye[iT]/(Omega*ClSI) << '\n';
		ofs.close();
		
		//Debye DOS:
		ofs.open("phDOSDebye.dat");
		ofs << "#omega[eV] phDOSDebye[eV^-1]\n";
		for(size_t ie=1; ie<dos.out.size(); ie++)
			ofs << ie*domega/eV << '\t'
				<< dosDebyeArr[ie]*eV << '\n';
	}
	
	fw.free();
	FeynWann::finalize();
}
