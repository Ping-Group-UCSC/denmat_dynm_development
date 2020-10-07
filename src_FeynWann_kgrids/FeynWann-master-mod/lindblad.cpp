/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman, Adela Habib

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

#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <commands/command.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "LindbladFile.h"
#include "Integrator.h"
#include <core/Units.h>

inline matrix dot(const matrix* P, vector3<complex> pol)
{	return pol[0]*P[0] + pol[1]*P[1] + pol[2]*P[2];
}

static const double degeneracyThreshold = 1e-5; //!< currently used only for spin-density calculation in report()

//Lindblad initialization, time evolution and measurement operators using FeynWann callback
struct Lindblad : public Integrator<DM1>
{	
	int stepID; //current time and reporting step number
	
	const double dmu, T, invT; //!< Fermi level position relative to neutral value / VBM, and temperature
	const double pumpOmega, pumpA0, pumpTau; const vector3<complex> pumpPol; const bool pumpEvolve; //!< pump parameters
	const double omegaMin, domega, omegaMax; const int nomega; //!< probe frequency grid
	const double tau; const std::vector<vector3<complex>> pol; //!< probe parameters
	const double dE; //!< energy resolution for distribution functions
	matrix3<> Gvec, GGT;
	vector3<> K, Kp;
	
	const bool ePhEnabled; //!< whether e-ph coupling is enabled
	const bool verbose; //!< whether to print more detailed stats during evolution
	bool spinorial; //!< whether spin is available
	int spinWeight; //!< weight of spin in BZ integration
	matrix3<> R; double Omega; //!< lattice vectors and unit cell volume
	
	size_t nk, nkTot; //!< number of selected k-points overall and original total k-points effectively used in BZ sampling
	size_t ikStart, ikStop, nkMine; //!< range and number of selected k-points on this process
	TaskDivision kDivision;
	inline bool isMine(size_t ik) const { return kDivision.isMine(ik); } //!< check if k-point index is local
	inline int whose(size_t ik) const { return kDivision.whose(ik); } //!< find out which process (in mpiWorld) this k-point belongs to

	struct State : LindbladFile::Kpoint
	{	int innerStop; //end of active inner window range (relative to outer window)
		diagMatrix rho0; //equilibrium / initial density matrix (diagonal)
		matrix pumpPD; //P matrix elements at pump polarization x energy conservation delta (D), but without A0 and time factor
	};
	std::vector<State> state; //!< all information read from lindbladInit output (e and e-ph properties) + extra local variables above
	std::vector<int> nInnerAll; //!< nInner for all k-points on all processes
	double Emin, Emax; //!< energy range of active space across all k (for spin and number density output)
	
	Lindblad(double dmu, double T, double pumpOmega, double pumpA0, double pumpTau, vector3<complex> pumpPol, bool pumpEvolve,
		double omegaMin, double omegaMax, double domega, double tau, std::vector<vector3<complex>> pol, double dE,
		bool ePhEnabled, bool verbose)
	: stepID(0),
		dmu(dmu), T(T), invT(1./T),
		pumpOmega(pumpOmega), pumpA0(pumpA0), pumpTau(pumpTau), pumpPol(pumpPol), pumpEvolve(pumpEvolve),
		omegaMin(omegaMin), domega(domega), omegaMax(omegaMax), nomega(1+int(round((omegaMax-omegaMin)/domega))),
		tau(tau), pol(pol), dE(dE), ePhEnabled(ePhEnabled), verbose(verbose),
		Emin(+DBL_MAX), Emax(-DBL_MAX),
		K(1. / 3, 1. / 3, 0), Kp(-1. / 3, -1. / 3, 0)
	{
	}
	
	
	//---- Flat density matrix storage and access functions ----
	DM1 rho; //!< flat array of density matrices of all k stored on this process
	std::vector<size_t> rhoOffset; //!< array of offsets into process's rho for each k
	std::vector<size_t> rhoSize; //!< total size of rho on each process
	
	//Get an NxN complex Hermitian matrix from a real array of length N^2
	inline matrix getRho(const double* rhoData, int N) const
	{	matrix out(N, N); complex* outData = out.data();
		for(int i=0; i<N; i++)
			for(int j=0; j<=i; j++)
			{	int i1 = i+N*j, i2 = j+N*i;
				if(i==j)
					outData[i1] = rhoData[i1];
				else
					outData[i2] = (outData[i1] = complex(rhoData[i1],rhoData[i2])).conj();
			}
		return out;
	}
	
	//Accumulate a diagonal matrix to a real array of length N^2
	inline void accumRho(const diagMatrix& in, double* rhoData) const
	{	const int N = in.nRows();
		for(int i=0; i<N; i++)
		{	*(rhoData) += in[i];
			rhoData += (N+1); //advance to next diagonal entry
		}
	}
	
	//Accumulate an NxN matrix and its Hermitian conjugate to a real array of length N^2
	inline void accumRhoHC(const matrix& in, double* rhoData) const
	{	const complex* inData = in.data();
		const int N = in.nRows();
		for(int i=0; i<N; i++)
			for(int j=0; j<=i; j++)
			{	int i1 = i+N*j, i2 = j+N*i;
				if(i==j)
					rhoData[i1] += 2*inData[i1].real();
				else
				{	complex hcSum = inData[i1] + inData[i2].conj();
					rhoData[i1] += hcSum.real();
					rhoData[i2] += hcSum.imag();
				}
			}
	}
	
	//--------- Initialize -------------
	void initialize()
	{
		//Read header and check parameters:
		MPIUtil::File fp;
		mpiWorld->fopenRead(fp, "ldbd.dat");
		LindbladFile::Header h; h.read(fp, mpiWorld);
		if(dmu<h.dmuMin or dmu>h.dmuMax)
			die("dmu = %lg eV is out of range [ %lg , %lg ] eV specified in lindbladInit.\n", dmu/eV, h.dmuMin/eV, h.dmuMax/eV);
		if(T > h.Tmax)
			die("T = %lg K is larger than Tmax = %lg K specified in lindbladInit.\n", T/Kelvin, h.Tmax/Kelvin);
		if(pumpOmega > h.pumpOmegaMax)
			die("pumpOmega = %lg eV is larger than pumpOmegaMax = %lg eV specified in lindbladInit.\n", pumpOmega/eV, h.pumpOmegaMax/eV);
		if(omegaMax > h.probeOmegaMax)
			die("omegaMax = %lg eV is larger than probeOmegaMax = %lg eV specified in lindbladInit.\n", omegaMax/eV, h.probeOmegaMax/eV);
		nk = h.nk;
		nkTot = h.nkTot;
		spinorial = h.spinorial;
		spinWeight = h.spinWeight;
		R = h.R; Omega = fabs(det(R));
		Gvec = (2.*M_PI)*inv(R);
		GGT = Gvec * (~Gvec);
		if(ePhEnabled != h.ePhEnabled)
			die("ePhEnabled = %s differs from the mode specified in lindbladInit.\n", boolMap.getString(ePhEnabled));
		
		//Read k-point offsets:
		std::vector<size_t> byteOffsets(h.nk);
		mpiWorld->freadData(byteOffsets, fp);
		
		//Divide k-points between processes:
		kDivision.init(nk, mpiWorld);
		kDivision.myRange(ikStart, ikStop);
		nkMine = ikStop-ikStart;
		state.resize(nkMine);
		nInnerAll.resize(nk);
		
		//Read k-point info and initialize states:
		mpiWorld->fseek(fp, byteOffsets[ikStart], SEEK_SET);
		for(size_t ikMine=0; ikMine<nkMine; ikMine++)
		{	State& s = state[ikMine];
			
			//Read base info from LindbladFile:
			((LindbladFile::Kpoint&)s).read(fp, mpiWorld, h);
			nInnerAll[ikStart+ikMine] = s.nInner;
			
			//Initialize extra quantities in state:
			s.innerStop = s.innerStart + s.nInner;
			//--- Active energy range:
			Emin = std::min(Emin, s.E[s.innerStart]);
			Emax = std::max(Emax, s.E[s.innerStop-1]);
			//--- Pump matrix elements with energy conservation
			s.pumpPD = dot(s.P, pumpPol)(0,s.nInner, s.innerStart,s.innerStop); //restrict to inner active
			double normFac = sqrt(pumpTau/sqrt(M_PI));
			complex* PDdata = s.pumpPD.data();
			for(int b2=s.innerStart; b2<s.innerStop; b2++)
				for(int b1=s.innerStart; b1<s.innerStop; b1++)
				{	//Multiply energy conservation:
					double tauDeltaE = pumpTau*(s.E[b1] - s.E[b2] - pumpOmega);
					*(PDdata++) *= normFac * exp(-0.5*tauDeltaE*tauDeltaE);
				}
			
			//Set initial occupations:
			s.rho0.resize(s.nInner);
			for(int b=0; b<s.nInner; b++)
				s.rho0[b] = fermi((s.E[b+s.innerStart]-dmu)*invT);
		}
		mpiWorld->fclose(fp);
		
		//Synchronize energy range:
		mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
		logPrintf("Electron energy grid from %lg eV to %lg eV with spacing %lg eV.\n", Emin/eV, Emax/eV, dE/eV);
		
		//Make nInner for all k available on all processes:
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			mpiWorld->bcast(nInnerAll.data()+kDivision.start(jProc),
				kDivision.stop(jProc)-kDivision.start(jProc), jProc);
		
		//Compute sizes of and offsets into flattened rho for all processes:
		rhoOffset.resize(nk);
		rhoSize.resize(mpiWorld->nProcesses());
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t jkStart = kDivision.start(jProc);
			size_t jkStop = kDivision.stop(jProc);
			size_t offset = 0; //start at 0 for each process's chunk
			for(size_t jk=jkStart; jk<jkStop; jk++)
			{	rhoOffset[jk] = offset;
				offset += nInnerAll[jk]*nInnerAll[jk];
			}
			rhoSize[jProc] = offset;
		}
		
		//Initialize rho:
		rho.resize(rhoSize[mpiWorld->iProcess()]);
		const State* sPtr = state.data();
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			accumRho(s.rho0, rho.data()+rhoOffset[ik]);
		}
		
		//Initialize A+ and A- for e-ph matrix elements if required:
		if(ePhEnabled)
		{	//Make inner-window energies available for all processes:
			std::vector<size_t> nInnerPrev(nk+1, 0);
			for(size_t ik=0; ik<nk; ik++)
				nInnerPrev[ik+1] = nInnerPrev[ik] + nInnerAll[ik];
			std::vector<double> Eall(nInnerPrev.back());
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = state[ik-ikStart];
				const double* Ei = &(s.E[s.innerStart]);
				std::copy(Ei, Ei+s.nInner, Eall.begin()+nInnerPrev[ik]);
			}
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	size_t iEstart = nInnerPrev[kDivision.start(jProc)];
				size_t iEstop = nInnerPrev[kDivision.stop(jProc)];
				mpiWorld->bcast(&Eall[iEstart], iEstop-iEstart, jProc);
			}
			//Initialize A+ and A- for all matrix elements:
			for(State& s: state)
			{	const double* Ei = &(s.E[s.innerStart]);
				for(LindbladFile::GePhEntry& g: s.GePh)
				{	const double* Ej = &(Eall[nInnerPrev[g.jk]]);
					g.initA(Ei, Ej, T);
				}
			}
		}
	}
	
	//Calculate probe response at current rho (update this->imEps)
	diagMatrix calcImEps() const
	{	static StopWatch watch("Lindblad::calcImEps");
		size_t nImEps = pol.size() * nomega;
		if(nImEps==0) return diagMatrix(); //no probe specified
		watch.start();
		diagMatrix imEps(nImEps);
		//Collect contributions from each k at this process:
		const State* sPtr = state.data();
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			const matrix rhoCurSub = getRho(rho.data()+rhoOffset[ik], s.nInner);
			//Expand density matrix:
			matrix rhoCur = zeroes(s.nOuter, s.nOuter);
			if(s.innerStart) rhoCur.set(0,s.innerStart, 0,s.innerStart, eye(s.innerStart));
			rhoCur.set(s.innerStart,s.innerStop, s.innerStart,s.innerStop, rhoCurSub);
			matrix rhoBar = eye(s.nOuter) - rhoCur; //1-rho
			//Expand probe matrix elements:
			std::vector<matrix> Ppol(pol.size(), zeroes(s.nOuter, s.nOuter));
			for(int iDir=0; iDir<3; iDir++)
			{	//Expand Cartesian component:
				const matrix& PiSub = s.P[iDir]; //nInner x nOuter
				matrix Pi = zeroes(s.nOuter, s.nOuter);
				Pi.set(s.innerStart,s.innerStop, 0,s.nOuter, PiSub);
				Pi.set(0,s.nOuter, s.innerStart,s.innerStop, dagger(PiSub));
				//Update each polarization:
				for(int iPol=0; iPol<int(pol.size()); iPol++)
					Ppol[iPol] += pol[iPol][iDir] * Pi;
			}
			//Probe response:
			for(int iomega=0; iomega<nomega; iomega++)
			{	double omega = omegaMin + iomega*domega;
				double prefac = (4*std::pow(M_PI,2)*spinWeight)/(nkTot * Omega * std::pow(std::max(omega, 1./tau), 3));
				//Energy conservation factors for all pair of bands at this frequency:
				std::vector<double> delta(s.nOuter*s.nOuter);
				double* deltaData = delta.data();
				double normFac = sqrt(tau/sqrt(M_PI));
				for(int b2=0; b2<s.nOuter; b2++)
					for(int b1=0; b1<s.nOuter; b1++)
					{	double tauDeltaE = tau*(s.E[b1] - s.E[b2] - omega);
						*(deltaData++) = normFac * exp(-0.5*tauDeltaE*tauDeltaE);
					}
				//Loop over polarizations:
				for(int iPol=0; iPol<int(pol.size()); iPol++)
				{	//Multiply matrix elements with energy conservation:
					matrix P = Ppol[iPol];
					eblas_zmuld(P.nData(), delta.data(),1, P.data(),1); //P-
					matrix Pdag = dagger(P); //P+
					//Loop over directions of excitations:
					diagMatrix deltaRhoDiag(s.nOuter);
					for(int s=-1; s<=+1; s+=2)
					{	deltaRhoDiag += diag(rhoBar*P*rhoCur*Pdag - Pdag*rhoBar*P*rhoCur);
						std::swap(P, Pdag); //P- <--> P+
					}
					imEps[iPol*nomega+iomega] += prefac * dot(s.E, deltaRhoDiag);
				}
			}
		}
		//Accumulate contributions from all processes on head:
		mpiWorld->reduceData(imEps, MPIUtil::ReduceSum);
		watch.stop();
		return imEps;
	}
	
	//Write current imEps to plain-text file:
	void writeImEps(string fname, const diagMatrix& imEps) const{
		if (mpiWorld->isHead()){
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "#omega[eV]");
			for (int iPol = 0; iPol<int(pol.size()); iPol++)
				fprintf(fp, " ImEps");
			fprintf(fp, "\n");
			for (int iomega = 0; iomega<nomega; iomega++){
				double omega = omegaMin + iomega*domega;
				fprintf(fp, "%14.7le ", omega / eV);
				for (int iPol = 0; iPol<int(pol.size()); iPol++)
					fprintf(fp, "%14.7le ", imEps[iPol*nomega + iomega]);
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
		/*
		{	ofstream ofs(fname);
			ofs << "#omega[eV]";
			for(int iPol=0; iPol<int(pol.size()); iPol++)
				ofs << " ImEps" << (iPol+1);
			ofs << "\n";
			for(int iomega=0; iomega<nomega; iomega++)
			{	double omega = omegaMin + iomega*domega;
				ofs << omega/eV;
				for(int iPol=0; iPol<int(pol.size()); iPol++)
					ofs << '\t' << imEps[iPol*nomega+iomega];
				ofs << '\n';
			}
		}
		*/
	}
	
	//Apply pump using perturbation theory (instantly go from before to after pump, skipping time evolution)
	void applyPump()
	{	static StopWatch watch("Lindblad::applyPump"); 
		if(pumpEvolve) return; //only use this function when perturbing instantly
		watch.start();
		const State* sPtr = state.data();
		//Perturb each k separately:
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			const matrix rhoCur = getRho(rho.data()+rhoOffset[ik], s.nInner);
			matrix rhoBar = eye(s.nInner) - rhoCur; //1-rho
			//Compute and apply perturbation:
			matrix P = s.pumpPD; //P-
			matrix Pdag = dagger(P); //P+
			matrix deltaRho;
			for(int s=-1; s<=+1; s+=2)
			{	deltaRho += rhoBar*P*rhoCur*Pdag - Pdag*rhoBar*P*rhoCur;
				std::swap(P, Pdag); //P- <--> P+
			}
			accumRhoHC((M_PI*pumpA0*pumpA0) * deltaRho, rho.data()+rhoOffset[ik]);
		}
		watch.stop();
	}
	
	//Time evolution operator returning drho/dt
	DM1 compute(double t, const DM1& rho)
	{	static StopWatch watchPump("Lindblad::compute::Pump");
		static StopWatch watchEph("Lindblad::compute::ePh");
		DM1 rhoDot(rho.size(), 0.);
		//Pump contribution:
		if(pumpEvolve)
		{	watchPump.start();
			double prefac = sqrt(M_PI)*pumpA0*pumpA0/pumpTau * exp(-(t*t)/(pumpTau*pumpTau));
			//Each k contributes separately:
			const State* sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				const matrix rhoCur = getRho(rho.data()+rhoOffset[ik], s.nInner);
				const matrix rhoBar = eye(s.nInner) - rhoCur; //1-rho
				//Compute and apply perturbation:
				matrix P = s.pumpPD; //P-
				matrix Pdag = dagger(P); //P+
				matrix rhoDotCur = zeroes(s.nInner, s.nInner);
				for(int s=-1; s<=+1; s+=2)
				{	rhoDotCur += rhoBar*P*rhoCur*Pdag - Pdag*rhoBar*P*rhoCur;
					std::swap(P, Pdag); //P- <--> P+
				}
				accumRhoHC(prefac*rhoDotCur, rhoDot.data()+rhoOffset[ik]);
			}
			watchPump.stop();
		}
		
		//E-ph relaxation contribution:
		if(ePhEnabled)
		{	watchEph.start();
			const double prefac = M_PI/nkTot;
			//Loop over process poviding other k data:
			int iProc = mpiWorld->iProcess(); //current process
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	//Make data from jProc available:
				DM1 rho_j, rhoDot_j(rhoSize[jProc]);
				if(jProc==iProc)
					rho_j = rho;
				else
					rho_j.resize(rhoSize[jProc]);
				mpiWorld->bcastData(rho_j, jProc);
				size_t jkStart = kDivision.start(jProc);
				size_t jkStop = kDivision.stop(jProc);
				//Loop over rho1 local to each process:
				const State* sPtr = state.data();
				for(size_t ik1=ikStart; ik1<ikStop; ik1++)
				{	const State& s = *(sPtr++);
					const int& nInner1 = nInnerAll[ik1];
					const matrix rho1 = getRho(rho.data()+rhoOffset[ik1], nInner1);
					const matrix rho1bar = eye(nInner1) - rho1;
					matrix rho1dot = zeroes(nInner1, nInner1);
					//Find first entry of GePh whose partner is on jProc (if any):
					std::vector<LindbladFile::GePhEntry>::const_iterator g = std::lower_bound(s.GePh.begin(), s.GePh.end(), jkStart);
					while(g != s.GePh.end())
					{	if(g->jk >= jkStop) break;
						const size_t& ik2 = g->jk;
						const int& nInner2 = nInnerAll[ik2];
						const matrix rho2 = getRho(rho_j.data()+rhoOffset[ik2], nInner2);
						const matrix rho2bar = eye(nInner2) - rho2;
						matrix rho2dot = zeroes(nInner2, nInner2);
						//Loop over all connections to the same partner k:
						while((g != s.GePh.end()) and (g->jk == ik2))
						{	//Phonon occupation factor:
// 							double omegaPhByT = g->omegaPh/T;
// 							double nPh = bose(std::max(1e-3, omegaPhByT));
// 							rho1dot += (prefac*nPh) * (rho1bar * SMSdag(g->G, rho2)) - (SMSdag(g->G, rho2bar) * rho1) * (prefac*(nPh+1)); //+ h.c. added together below
// 							rho2dot += (prefac*(nPh+1)) * (SdagMS(g->G, rho1) * rho2bar) - (rho2 * SdagMS(g->G, rho1bar)) * (prefac*nPh); //+ h.c. added together below
							rho1dot += prefac * (rho1bar * SMSdag(g->Am, rho2) - SMSdag(g->Ap, rho2bar) * rho1); //+ h.c. added together below
							rho2dot += prefac * (SdagMS(g->Ap, rho1) * rho2bar - rho2 * SdagMS(g->Am, rho1bar)); //+ h.c. added together below
							//Move to next element:
							g++;
						}
						//Accumulate rho2 gradients:
						accumRhoHC(rho2dot, rhoDot_j.data()+rhoOffset[ik2]);
					}
					//Accumulate rho1 gradients:
					accumRhoHC(rho1dot, rhoDot.data()+rhoOffset[ik1]);
				}
				//Collect remote contributions:
				mpiWorld->reduceData(rhoDot_j, MPIUtil::ReduceSum, jProc);
				if(jProc==iProc) axpy(1., rhoDot_j, rhoDot);
			}
			watchEph.stop();
		}
		
		if(verbose)
		{	//Report current statistics:
			double rhoDotMax = 0., rhoEigMin = +DBL_MAX, rhoEigMax = -DBL_MAX;
			const State* sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				//max(rhoDot)
				const matrix rhoDotCur = getRho(rhoDot.data()+rhoOffset[ik], s.nInner);
				rhoDotMax = std::max(rhoDotMax, rhoDotCur.data()[cblas_izamax(rhoDotCur.nData(), rhoDotCur.data(), 1)].abs());
				//eig(rho):
				const matrix rhoCur = getRho(rho.data()+rhoOffset[ik], s.nInner);
				matrix V; diagMatrix f;
				rhoCur.diagonalize(V, f);
				rhoEigMin = std::min(rhoEigMin, f.front());
				rhoEigMax = std::max(rhoEigMax, f.back());
			}
			mpiWorld->reduce(rhoDotMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(rhoEigMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(rhoEigMin, MPIUtil::ReduceMin);
			logPrintf("\n\tComputed at t[fs]: %lg  max(rhoDot): %lg rhoEigRange: [ %lg %lg ] ",
				t/fs, rhoDotMax, rhoEigMin, rhoEigMax); logFlush();
		}
		else logPrintf("(t[fs]: %lg) ", t/fs);
		logFlush();
		return rhoDot;
	}
	
	//Print / dump quantities at each checkpointed step
	static inline vector3<> wrap(const vector3<>& x)
	{
		vector3<> result = x;
		for (int dir = 0; dir<3; dir++)
			result[dir] -= floor(0.5 + result[dir]);
		return result;
	}
	inline bool isKvalley(vector3<> k) const
	{
		return GGT.metric_length_squared(wrap(K - k))
			< GGT.metric_length_squared(wrap(Kp - k));
	}
	void report(double t, const DM1& rho) const
	{	static StopWatch watch("Lindblad::report"); watch.start();
		ostringstream ossID; ossID << stepID;
		//Compute total energy and distributions:
		int nDist = spinorial ? 4 : 1; //number distribution only, or also spin distribution
		std::vector<Histogram> dist(nDist, Histogram(Emin, dE, Emax));
		Histogram distValley(Emin, dE, Emax);
		const double prefac = spinWeight*(1./nkTot); //BZ integration weight
		double Etot = 0., dfMax = 0.;
		const State* sPtr = state.data();
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			bool isK = isKvalley(s.k); double wKvalley = isK ? 1. : -1.;
			const matrix rhoCur = getRho(rho.data()+rhoOffset[ik], s.nInner);
			matrix drho = rhoCur - s.rho0;
			//Energy and distribution:
			const complex* drhoData = drho.data();
			for(int b=0; b<s.nInner; b++)
			{	double weight = prefac * drhoData->real();
				const double& Ecur = s.E[b+s.innerStart];
				Etot += weight * Ecur;
				dfMax = std::max(dfMax, fabs(drhoData->real()));
				dist[0].addEvent(Ecur, weight);
				distValley.addEvent(Ecur, weight * wKvalley);
				drhoData += (s.nInner+1); //advance to next diagonal entry
			}
			//Spin distribution of available:
			if(spinorial)
			{	const complex* drhoData = drho.data();
				vector3<const complex*> Sdata; for(int k=0; k<3; k++) Sdata[k] = s.S[k].data();
				for(int b2=0; b2<s.nInner; b2++)
				{	int i = b2*s.nInner; //offset into data
					const double& E2 = s.E[b2+s.innerStart];
					for(int b1=0; b1<=b2; b1++) //use Hermitian symmetry
					{	complex weight = ((b1==b2 ? 1 : 2) * prefac) * drhoData[i];
						const double& E1 = s.E[b1+s.innerStart];
						//if(fabs(E1-E2) < degeneracyThreshold)
						//{	//Precalculate histogram position:
						//	double E = 0.5*(E1+E2);
						int iEvent; double tEvent;
						if (dist[1].eventPrecalc(E1, iEvent, tEvent))
						{       //Collect spin densities:
							for (int k = 0; k<3; k++)
								//dist[k+1].addEventPrecalc(iEvent, tEvent, (weight * Sdata[k][i]).real());
								dist[k + 1].addEventPrecalc(iEvent, tEvent, (weight * conj(Sdata[k][i])).real());
						}
						//}
						//Advance to next entry of Hermitian matrix:
						i++;
					}
				}
			}
		}
		mpiWorld->reduce(Etot, MPIUtil::ReduceSum);
		mpiWorld->reduce(dfMax, MPIUtil::ReduceMax);
		for(Histogram& h: dist) h.reduce(MPIUtil::ReduceSum);
		distValley.reduce(MPIUtil::ReduceSum);
		if(mpiWorld->isHead())
		{	//Report step ID and energy:
			logPrintf("Integrate: Step: %4d   t[fs]: %6.1lf   Etot[eV]: %.6lf   dfMax: %.3lg\n", stepID, t/fs, Etot/eV, dfMax);
			//Save distribution functions:
			string fname = "dist." + ossID.str();
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "#E-mu/VBM[eV] n[eV^-1]");
			if (spinorial)
				fprintf(fp, "Sx[eV^-1] Sy[eV^-1] Sz[eV^-1] Valley Pol[eV^-1]");
			fprintf(fp, "\n");
			for (int iE = 0; iE<dist[0].nE; iE++)
			{
				double E = Emin + iE*dE;
				fprintf(fp, "%14.7le ", E / eV);
				for (int iDist = 0; iDist<nDist; iDist++)
					fprintf(fp, "%14.7le ", dist[iDist].out[iE] * eV);
				fprintf(fp, "%14.7le ", distValley.out[iE] * eV);
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
		watch.stop();
		//Probe responses if present:
		diagMatrix imEps = calcImEps();
		if(imEps.size())
			writeImEps("imEps."+ossID.str(), imEps);
		((Lindblad*)this)->stepID++;
	}
};

inline void print(FILE* fp, const vector3<complex>& v, const char* format="%lg ")
{	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].real()); std::fprintf(fp, "] + 1j*");
	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].imag()); std::fprintf(fp, "]\n");
}
inline vector3<complex> normalize(const vector3<complex>& v) { return v * (1./sqrt(v[0].norm() + v[1].norm() + v[2].norm())); }

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Lindblad dynamics in an ab initio Wannier basis");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- doping / temperature
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional: shift in fermi level from neutral value / VBM in eV (default: 0)
	const double T = inputMap.get("T") * Kelvin; //temperature in Kelvin (ambient phonon T = initial electron T)
	//--- pump
	const string pumpMode = inputMap.getString("pumpMode"); //must be Perturb or Evolve
	if(pumpMode!="Evolve" and pumpMode!="Perturb")
		die("\npumpMode must be 'Evolve' or 'Perturb'\n");
	const double pumpOmega = inputMap.get("pumpOmega") * eV; //pump frequency in eV
	const double pumpA0 = inputMap.get("pumpA0"); //pump pulse amplitude / intensity (Units TBD)
	const double pumpTau = inputMap.get("pumpTau")*fs; //Gaussian pump pulse width in fs
	const vector3<complex> pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes
	const double omegaMin = inputMap.get("omegaMin") * eV; //start of frequency grid for probe response
	const double omegaMax = inputMap.get("omegaMax") * eV; //end of frequency grid for probe response
	const double domega = inputMap.get("domega") * eV; //frequency resolution for probe calculation
	const double tau = inputMap.get("tau") * fs; //Gaussian probe pulse width in fs
	std::vector<vector3<complex>> pol;
	while(true)
	{	int iPol = int(pol.size())+1;
		ostringstream oss; oss << iPol;
		string polName = oss.str();
		vector3<> polRe = inputMap.getVector("polRe"+polName, vector3<>(0.,0.,0.)); //Real part of polarization
		vector3<> polIm = inputMap.getVector("polIm"+polName, vector3<>(0.,0.,0.)); //Imag part of polarization
		if(polRe.length_squared() + polIm.length_squared() == 0.) break; //End of probe polarizations
		pol.push_back(normalize(complex(1,0)*polRe + complex(0,1)*polIm));
	}
	const double dE = inputMap.get("dE") * eV; //energy resolution for distribution functions
	const double dt = inputMap.get("dt") * fs; //time interval between reports
	const double tStop = inputMap.get("tStop") * fs; //stopping time for simulation
	const double tolAdapt = inputMap.get("tolAdapt", 1e-3);
	
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	if(ePhMode!="Off" and ePhMode!="DiagK")
		die("\nePhMode must be 'Off' or 'DiagK'\n");
	const bool ePhEnabled = (ePhMode != "Off");
	const string verboseMode = inputMap.getString("verbose"); //must be yes or no
	if(verboseMode!="yes" and verboseMode!="no")
		die("\nverboseMode must be 'yes' or 'no'\n");
	const bool verbose = (verboseMode=="yes");
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("T = %lg\n", T);
	logPrintf("pumpMode = %s\n", pumpMode.c_str());
	logPrintf("pumpOmega = %lg\n", pumpOmega);
	logPrintf("pumpA0 = %lg\n", pumpA0);
	logPrintf("pumpTau = %lg\n", pumpTau);
	logPrintf("pumpPol = "); print(globalLog, pumpPol);
	logPrintf("omegaMin = %lg\n", omegaMin);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("domega = %lg\n", domega);
	logPrintf("tau = %lg\n", tau);
	for(int iPol=0; iPol<int(pol.size()); iPol++)
	{	logPrintf("pol%d = ", iPol+1);
		print(globalLog, pol[iPol]);
	}
	logPrintf("dE = %lg\n", dE);
	logPrintf("dt = %lg\n", dt);
	logPrintf("tStop = %lg\n", tStop);
	logPrintf("tolAdapt = %lg\n", tolAdapt);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("verbose = %s\n", verboseMode.c_str());
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Create and initialize lindblad calculator:
	Lindblad lb(dmu, T,
		pumpOmega, pumpA0, pumpTau, pumpPol, (pumpMode=="Evolve"),
		omegaMin, omegaMax, domega, tau, pol, dE, ePhEnabled, verbose);
	lb.initialize();
	logPrintf("Initialization completed successfully at t[s]: %9.2lf\n\n", clock_sec());
	logFlush();
	
	if(pumpMode=="Perturb" and (not ePhEnabled))
	{	//Simple probe-pump-probe with no relaxation:
		lb.report(-dt, lb.rho);
		lb.applyPump();
		lb.report(0., lb.rho);
	}
	else
	{	double tStart = 0.;
		if(pumpMode=="Perturb")
		{	//Do an initial report akin to above and apply the pump:
			lb.report(-dt, lb.rho);
			lb.applyPump();
			tStart = 0.; //integrate will report at t=0 below, before evolving ePh relaxation
		}
		else
		{	//Set start time to a multiple of dt that covers pulse:
			tStart = -dt * ceil(5.*tau/dt);
		}
		//Evolve:
		lb.integrateAdaptive(lb.rho, tStart, tStop, tolAdapt, dt);
	}
	
	//Cleanup:
	FeynWann::finalize();
	return 0;
}
