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

struct CollectDOS
{	Histogram* dos;
	double weight;
	
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	CollectDOS& cd = *((CollectDOS*)params);
		for(const double& Ei: state.E)
			cd.dos->addEvent(Ei, cd.weight);
	}
};

int main(int argc, char** argv)
{	InitParams ip = FeynWann::initialize(argc, argv, "Electronic DOS and heat capacity");
	
	InputMap inputMap(ip.inputFilename);
	size_t nOffsets = inputMap.get("nOffsets");
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double Tmin = inputMap.get("Tmin") * Kelvin; //electron temperature grid start
	const double Tmax = inputMap.get("Tmax") * Kelvin; //electron temperature grid stop
	const double Tstep = inputMap.get("Tstep") * Kelvin; //electron temperature grid spacing
	FeynWannParams fwp(&inputMap);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %lu\n", nOffsets);
	logPrintf("dE = %lg\n", dE);
	logPrintf("Tmin = %lg\n", Tmin);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("Tstep = %lg\n", Tstep);
	fwp.printParams();
	
	//Initialize FeynWann:
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKpts = nOffsets * fw->eCountPerOffset();  
	logPrintf("Effectively sampled nKpts: %lu\n", nKpts);
	if(mpiWorld->isHead()) logPrintf("%lu electron k-mesh offsets parallelized over %d process groups.\n", nOffsets, mpiGroupHead->nProcesses());
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw = 0;
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");

	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Initialize temperature grid:
	std::vector<double> Tarr(int(ceil((Tmax-Tmin)/Tstep)));
	for(size_t iT=0; iT<Tarr.size(); iT++)
		Tarr[iT] = Tmin + Tstep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", Tarr.front()/Kelvin, Tarr.back()/Kelvin, Tarr.size());
	
	std::vector<std::shared_ptr<Histogram>> dosArr(fw->nSpins);
	for(int iSpin=0; iSpin<fw->nSpins; iSpin++)
	{	//Update FeynWann for spin channel if necessary:
		if(iSpin>0)
		{	fw = 0; //free memory from previous spin
			fwp.iSpin = iSpin;
			fw = std::make_shared<FeynWann>(fwp);
		}
		
		//Initialize energy grid:
		EnergyRange er = { DBL_MAX, -DBL_MAX };
		fw->eLoop(vector3<>(), EnergyRange::eProcess, &er);
		mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
		er.Emin = dE * (floor(er.Emin/dE) - 10); //add some margin and ensure grid contains 0
		er.Emax = dE * (ceil(er.Emax/dE) + 10);
		dosArr[iSpin] = std::make_shared<Histogram>(er.Emin, dE, er.Emax); //density of states for current spin channel
		Histogram& dos = *dosArr[iSpin];
		logPrintf("Initialized energy grid: %lg to %lg eV with %d points.\n", dos.Emin/eV, dos.Emax()/eV, dos.nE);
		
		logPrintf("\nCollecting DOS: "); logFlush();
		CollectDOS cd;
		cd.dos = &dos;
		cd.weight = fw->spinWeight*(1./nKpts);
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			//Process with a random offset:
			vector3<> k0 = fw->randomVector(mpiGroup); //must be constant across group
			fw->eLoop(k0, CollectDOS::eProcess, &cd);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		dos.allReduce(MPIUtil::ReduceSum);
	}
	
	//Output DOS, combining spin channels if necessary
	if(fw->nSpins == 1)
		dosArr[0]->print("eDOS.dat", 1./eV, eV);
	else
	{	//Combined energy range:
		double Emin = DBL_MAX, Emax = -DBL_MAX;
		for(const auto& dos: dosArr)
		{	Emin = std::min(Emin, dos->Emin);
			Emax = std::max(Emax, dos->Emax());
		}
		//Convert DOS to combined energy grid and collect total:
		std::shared_ptr<Histogram> dosTot = std::make_shared<Histogram>(Emin, dE, Emax);
		logPrintf("\nCombined energy grid: %lg to %lg eV with %d points.\n", dosTot->Emin/eV, dosTot->Emax()/eV, dosTot->nE);
		for(auto& dos: dosArr)
		{	std::shared_ptr<Histogram> dosNew = std::make_shared<Histogram>(Emin, dE, Emax);
			int iOffset = round((dos->Emin - Emin)/dE);
			for(int i=0; i<dos->nE; i++)
			{	dosNew->out[i+iOffset] += dos->out[i];
				dosTot->out[i+iOffset] += dos->out[i];
			}
			std::swap(dos, dosNew);
		}
		//Output combined result along with individual ones:
		if(mpiWorld->isHead())
		{	ofstream ofs("eDOS.dat");
			double Escale = 1./eV, histScale=eV;
			for(int i=0; i<dosTot->nE; i++)
			{	ofs << (Emin+i*dE)*Escale << "\t" << dosTot->out[i]*histScale;
				for(const auto& dos: dosArr)
					ofs << "\t" << dos->out[i]*histScale;
				ofs << '\n';
			}
		}
		dosArr.push_back(dosTot); //add total DOS as last channel
	}
	const Histogram& dos = *dosArr.back(); //last channel is total DOS (irrespective of spin)
	
	//Calculate mu and Ce at each temperature:
	diagMatrix dmu(Tarr.size(), 0.), Ce(Tarr.size(), 0.);
	//--- check enough bands to contain Z:
	double Zmax = 0.;
	for(const double& g: dos.out)
		Zmax += dE * g;
	if(Zmax < fw->nElectrons)
		die("Current DOS can only support %lg electrons > %lg electrons specified.\n", Zmax, fw->nElectrons);
	int iTstart, iTstop; TaskDivision(Tarr.size(), mpiWorld).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double T = Tarr[iT], invT = 1./T;
		//Bisect for chemical potential:
		double& dmuCur = dmu[iT];
		double dmuMin = dos.Emin - 10*T;
		double dmuMax = dos.Emax() + 10*T;
		dmuCur = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*T;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(int ie=0; ie<dos.nE; ie++)
			{	double Ei = dos.Emin + ie*dE;
				double fi = fermi(invT*(Ei - dmuCur));
				nElectrons += dE * dos.out[ie] * fi;
			}
			((nElectrons>fw->nElectrons) ? dmuMax : dmuMin) = dmuCur;
			dmuCur = 0.5*(dmuMin + dmuMax);
		}
		//Calculate electronic specific heat:
		double& CeCur = Ce[iT];
		CeCur = 0.;
		for(int ie=0; ie<dos.nE; ie++)
		{	double Ei = dos.Emin + ie*dE;
			double x = invT*(Ei-dmuCur);
			double dfdT = fermiPrime(x) * (-x*invT);
			CeCur += dE * Ei * dos.out[ie] * dfdT;
		}
	}
	mpiWorld->allReduceData(dmu, MPIUtil::ReduceSum);
	mpiWorld->allReduceData(Ce, MPIUtil::ReduceSum);
	
	//Write to file
	if(mpiWorld->isHead())
	{	const double CeSI = Joule/(Kelvin*pow(meter,3));
		ofstream ofs("Ce.dat");
		ofs << "#T[K] Ce[J/m^3K] dmu[eV]\n";
		for(size_t iT=0; iT<Tarr.size(); iT++)
			ofs << Tarr[iT]/Kelvin << '\t'
				<< Ce[iT]/(fw->Omega*CeSI) << '\t'
				<< dmu[iT]/eV << '\n';
	}
	
	fw = 0;
	FeynWann::finalize();
}
