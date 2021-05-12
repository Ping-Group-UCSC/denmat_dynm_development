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

//Construct identity - X:
inline matrix bar(const matrix& X)
{	matrix Xbar(X);
	complex* XbarData = Xbar.data();
	for(int j=0; j<X.nCols(); j++)
		for(int i=0; i<X.nRows(); i++)
		{	(*XbarData) = (i==j ? 1. : 0.) - (*XbarData);
			XbarData++;
		}
	return Xbar;
}

//Lindblad initialization, time evolution and measurement operators using FeynWann callback
struct Lindblad : public Integrator<DM1>
{	
	int stepID; //current time and reporting step number
	
	const double dmu, T, invT; //!< Fermi level position relative to neutral value / VBM, and temperature
	const double pumpOmega, pumpA0, pumpTau; const vector3<complex> pumpPol; const bool pumpEvolve; //!< pump parameters
	const bool pumpBfield; const vector3<> pumpB; //pump parameters for Bfield mode
	const double omegaMin, domega, omegaMax; const int nomega; //!< probe frequency grid
	const double tau; const std::vector<vector3<complex>> pol; //!< probe parameters
	const double dE; //!< energy resolution for distribution functions
	
	const bool ePhEnabled; //!< whether e-ph coupling is enabled
	const double defectFraction; //!< defect fraction if present
	const bool verbose; //!< whether to print more detailed stats during evolution
	const string checkpointFile; //!< file name to save checkpoint data to
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
	
	std::vector<double> Eall; //!< inner window energies for all k (only needed and initialized when ePhEnabled)
	std::vector<size_t> nInnerPrev; //!< cumulative nInner for each k, which is the offset into the Eall array for each k
	double tPrev; //last time at which compute() was called; used internally to update e-ph operator phases
	
	Lindblad(double dmu, double T, 
		double pumpOmega, double pumpA0, double pumpTau, vector3<complex> pumpPol, bool pumpEvolve, bool pumpBfield, vector3<> pumpB,
		double omegaMin, double omegaMax, double domega, double tau, std::vector<vector3<complex>> pol, double dE,
		bool ePhEnabled, double defectFraction, bool verbose, string checkpointFile)
	: stepID(0),
		dmu(dmu), T(T), invT(1./T),
		pumpOmega(pumpOmega), pumpA0(pumpA0), pumpTau(pumpTau), pumpPol(pumpPol), pumpEvolve(pumpEvolve),
		pumpBfield(pumpBfield), pumpB(pumpB),
		omegaMin(omegaMin), domega(domega), omegaMax(omegaMax), nomega(1+int(round((omegaMax-omegaMin)/domega))),
		tau(tau), pol(pol), dE(dE), ePhEnabled(ePhEnabled), defectFraction(defectFraction),
		verbose(verbose), checkpointFile(checkpointFile),
		Emin(+DBL_MAX), Emax(-DBL_MAX), tPrev(0.)
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
	void initialize(string inFile)
	{
		//Read header and check parameters:
		MPIUtil::File fp;
		mpiWorld->fopenRead(fp, inFile.c_str());
		LindbladFile::Header h; h.read(fp, mpiWorld);
		if(dmu<h.dmuMin or dmu>h.dmuMax)
			die("dmu = %lg eV is out of range [ %lg , %lg ] eV specified in lindbladInit.\n", dmu/eV, h.dmuMin/eV, h.dmuMax/eV);
		if(T > h.Tmax)
			die("T = %lg K is larger than Tmax = %lg K specified in lindbladInit.\n", T/Kelvin, h.Tmax/Kelvin);
		if((not pumpBfield) and (pumpOmega > h.pumpOmegaMax))
			die("pumpOmega = %lg eV is larger than pumpOmegaMax = %lg eV specified in lindbladInit.\n", pumpOmega/eV, h.pumpOmegaMax/eV);
		if(omegaMax > h.probeOmegaMax)
			die("omegaMax = %lg eV is larger than probeOmegaMax = %lg eV specified in lindbladInit.\n", omegaMax/eV, h.probeOmegaMax/eV);
		nk = h.nk;
		nkTot = h.nkTot;
		spinorial = h.spinorial;
		spinWeight = h.spinWeight;
		R = h.R; Omega = fabs(det(R));
		if(ePhEnabled != h.ePhEnabled)
			die("ePhEnabled = %s differs from the mode specified in lindbladInit.\n", boolMap.getString(ePhEnabled));
		if(pumpBfield and (not spinorial))
			die("Bfield pump mode requires spin matrix elements from a spinorial calculation.\n");
		
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
			if(not pumpBfield)
			{	s.pumpPD = dot(s.P, pumpPol)(0,s.nInner, s.innerStart,s.innerStop); //restrict to inner active
				double normFac = sqrt(pumpTau/sqrt(M_PI));
				complex* PDdata = s.pumpPD.data();
				for(int b2=s.innerStart; b2<s.innerStop; b2++)
					for(int b1=s.innerStart; b1<s.innerStop; b1++)
					{	//Multiply energy conservation:
						double tauDeltaE = pumpTau*(s.E[b1] - s.E[b2] - pumpOmega);
						*(PDdata++) *= normFac * exp(-0.5*tauDeltaE*tauDeltaE);
					}
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
			nInnerPrev.assign(nk+1, 0);
			for(size_t ik=0; ik<nk; ik++)
				nInnerPrev[ik+1] = nInnerPrev[ik] + nInnerAll[ik];
			Eall.resize(nInnerPrev.back());
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
			{	for(LindbladFile::GePhEntry& g: s.GePh)
				{	g.G.init(s.nInner, nInnerAll[g.jk]);
					g.initA(T, defectFraction);
				}
			}
		}
		logPrintf("\n"); logFlush();
	}
	
	//Calculate probe response at current rho (update this->imEps)
	diagMatrix calcImEps(double t) const
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
			matrix rhoBar(bar(rhoCur)); //1-rho
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
				//Energy conservation and phase factors for all pair of bands at this frequency:
				std::vector<complex> delta(s.nOuter*s.nOuter);
				complex* deltaData = delta.data();
				double normFac = sqrt(tau/sqrt(M_PI));
				for(int b2=0; b2<s.nOuter; b2++)
					for(int b1=0; b1<s.nOuter; b1++)
					{	double tauDeltaE = tau*(s.E[b1] - s.E[b2] - omega);
						*(deltaData++) = normFac * exp(-0.5*tauDeltaE*tauDeltaE) * cis(t*(s.E[b1]-s.E[b2]));
					}
				//Loop over polarizations:
				for(int iPol=0; iPol<int(pol.size()); iPol++)
				{	//Multiply matrix elements with energy conservation:
					matrix P = Ppol[iPol];
					eblas_zmul(P.nData(), delta.data(),1, P.data(),1); //P-
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
	void writeImEps(string fname, const diagMatrix& imEps) const
	{	if(mpiWorld->isHead())
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
			if(pumpBfield)
			{	//Construct Hamiltonian including magnetic field contribution:
				matrix Htot(s.E(s.innerStart, s.innerStart+s.nInner));
				for(int iDir=0; iDir<3; iDir++) //Add Zeeman Hamiltonian
					Htot -= pumpB[iDir] * s.S[iDir];
				//Set rho to Fermi function of this perturbed Hamiltonian:
				diagMatrix Epert; matrix Vpert;
				Htot.diagonalize(Vpert, Epert);
				diagMatrix fPert(s.nInner);
				for(int b=0; b<s.nInner; b++)
					fPert[b] = fermi((Epert[b]-dmu)*invT);
				matrix rhoPert = Vpert * fPert * dagger(Vpert);
				accumRhoHC(0.5*(rhoPert-rhoCur), rho.data()+rhoOffset[ik]);
			}
			else
			{	matrix rhoBar(bar(rhoCur)); //1-rho
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
		}
		watch.stop();
	}
	
	//Time evolution operator returning drho/dt
	DM1 compute(double t, const DM1& rho)
	{	static StopWatch watchPump("Lindblad::compute::Pump");
		static StopWatch watchEph("Lindblad::compute::ePh");
		static StopWatch watchEphInner("Lindblad::compute::ePhInner");
		
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
				const matrix rhoBar(bar(rhoCur)); //1-rho
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
					const matrix rho1bar(bar(rho1));
					matrix rho1dot = zeroes(nInner1, nInner1);
					const double* E1 = &(s.E[s.innerStart]);
					//Find first entry of GePh whose partner is on jProc (if any):
					std::vector<LindbladFile::GePhEntry>::const_iterator g = std::lower_bound(s.GePh.begin(), s.GePh.end(), jkStart);
					while(g != s.GePh.end())
					{	if(g->jk >= jkStop) break;
						const size_t& ik2 = g->jk;
						const int& nInner2 = nInnerAll[ik2];
						const matrix rho2 = getRho(rho_j.data()+rhoOffset[ik2], nInner2);
						const matrix rho2bar(bar(rho2));
						matrix rho2dot = zeroes(nInner2, nInner2);
						const double* E2 = &(Eall[nInnerPrev[ik2]]);
						//Loop over all connections to the same partner k:
						watchEphInner.start();
						while((g != s.GePh.end()) and (g->jk == ik2))
						{	//Update phases for A+ and A- (interaction picture):
							SparseMatrix& Am = (SparseMatrix&)g->Am;
							SparseMatrix& Ap = (SparseMatrix&)g->Ap;
							for(SparseMatrix::iterator sm=Am.begin(),sp=Ap.begin(); sm!=Am.end(); sm++,sp++)
							{	double deltaE = E1[sm->i] - E2[sm->j];
								complex phase = cis(deltaE*(t-tPrev));
								sm->val *= phase;
								sp->val *= phase;
							}
							//Contributions to rho1dot: (+ h.c. added together by accumRhoHC)
							axpyMSMS<false,true>(+prefac, rho1bar, Am, rho2, Am, rho1dot);
							axpySMSM<false,true>(-prefac, Ap, rho2bar, Ap, rho1, rho1dot);
							//Contributions to rho2dot: (+ h.c. added together by accumRhoHC)
							axpySMSM<true,false>(+prefac, Ap, rho1, Ap, rho2bar, rho2dot);
							axpyMSMS<true,false>(-prefac, rho2, Am, rho1bar, Am, rho2dot);
							//Move to next element:
							g++;
						}
						watchEphInner.stop();
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
		
		tPrev = t;
		return rhoDot;
	}
	
	//Print / dump quantities at each checkpointed step
	void report(double t, const DM1& rho) const
	{	static StopWatch watch("Lindblad::report"); watch.start();
		ostringstream ossID; ossID << stepID;
		//Compute total energy and distributions:
		int nDist = spinorial ? 4 : 1; //number distribution only, or also spin distribution
		std::vector<Histogram> dist(nDist, Histogram(Emin, dE, Emax));
		const double prefac = spinWeight*(1./nkTot); //BZ integration weight
		double Etot = 0., dfMax = 0.; vector3<> Stot;
		const State* sPtr = state.data();
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
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
				drhoData += (s.nInner+1); //advance to next diagonal entry
			}
			//Spin distribution (if available):
			if(spinorial)
			{	const complex* drhoData = drho.data();
				vector3<const complex*> Sdata; for(int k=0; k<3; k++) Sdata[k] = s.S[k].data();
				std::vector<vector3<>> Sband(s.nInner); //spin expectation by band S_b := sum_a S_ba drho_ab
				const double* Einner = s.E.data() + s.innerStart;
				for(int b2=0; b2<s.nInner; b2++)
				{	for(int b1=0; b1<s.nInner; b1++)
					{	complex weight = prefac * (*(drhoData++)).conj() * cis((Einner[b1]-Einner[b2])*t);
						for(int iDir=0; iDir<3; iDir++)
							Sband[b2][iDir] += (weight * (*(Sdata[iDir]++))).real();
					}
					Stot += Sband[b2];
				}
				//Collect distribution based on per-band spin:
				for(int b=0; b<s.nInner; b++)
				{	const double& E = s.E[b+s.innerStart];
					int iEvent; double tEvent;
					if(dist[1].eventPrecalc(E, iEvent, tEvent))
					{	for(int iDir=0; iDir<3; iDir++)
							dist[iDir+1].addEventPrecalc(iEvent, tEvent, Sband[b][iDir]);
					}
				}
			}
		}
		mpiWorld->reduce(Etot, MPIUtil::ReduceSum);
		mpiWorld->reduce(Stot, MPIUtil::ReduceSum);
		mpiWorld->reduce(dfMax, MPIUtil::ReduceMax);
		for(Histogram& h: dist) h.reduce(MPIUtil::ReduceSum);
		if(mpiWorld->isHead())
		{	//Report step ID and energy:
			logPrintf("Integrate: Step: %4d   t[fs]: %6.1lf   Etot[eV]: %.6lf   dfMax: %.4lf", stepID, t/fs, Etot/eV, dfMax);
			if(spinorial) logPrintf("   S: [ %16.15lg %16.15lg %16.15lg ]", Stot[0],  Stot[1],  Stot[2]);
			logPrintf("\n"); logFlush();
			//Save distribution functions:
			ofstream ofs("dist."+ossID.str());
			ofs << "#E-mu/VBM[eV] n[eV^-1]";
			if(spinorial)
				ofs << "Sx[eV^-1] Sy[eV^-1] Sz[eV^-1]";
			ofs << "\n";
			for(int iE=0; iE<dist[0].nE; iE++)
			{	double E = Emin + iE*dE;
				ofs << E/eV;
				for(int iDist=0; iDist<nDist; iDist++)
					ofs << '\t' << dist[iDist].out[iE]*eV;
				ofs << '\n';
			}
		}
		//Write checkpoint file if needed:
		if(checkpointFile.length())
		{
			#ifdef MPI_SAFE_WRITE
			if(mpiWorld->isHead())
			{	FILE* fp = fopen(checkpointFile.c_str(), "w");
				fwrite(&stepID, sizeof(int), 1, fp);
				fwrite(&t, sizeof(double), 1, fp);
				//Data from head:
				fwrite(rho.data(), sizeof(double), rho.size(), fp);
				//Data from remaining processes:
				for(int jProc=1; jProc<mpiWorld->nProcesses(); jProc++)
				{	DM1 buf(rhoSize[jProc]);
					mpiWorld->recvData(buf, jProc, 0); //recv data to be written
					fwrite(buf.data(), sizeof(double), buf.size(), fp);
				}
				fclose(fp);
			}
			else mpiWorld->sendData(rho, 0, 0); //send to head for writing
			#else
			//Write in parallel using MPI I/O:
			MPIUtil::File fp; mpiWorld->fopenWrite(fp, checkpointFile.c_str());
			//--- Write current step and time as a header:
			if(mpiWorld->isHead())
			{	mpiWorld->fwrite(&stepID, sizeof(int), 1, fp);
				mpiWorld->fwrite(&t, sizeof(double), 1, fp);
			}
			//--- Move to location of this process's data:
			size_t offset = sizeof(int) + sizeof(double); //offset due to header
			for(int jProc=0; jProc<mpiWorld->iProcess(); jProc++)
				offset += sizeof(double) * rhoSize[jProc]; //offset due to data from previous processes
			mpiWorld->fseek(fp, offset, SEEK_SET);
			//--- Write this process's data:
			mpiWorld->fwrite(rho.data(), sizeof(double), rho.size(), fp);
			mpiWorld->fclose(fp);
			#endif
		}
		watch.stop();
		//Probe responses if present:
		diagMatrix imEps = calcImEps(t);
		if(imEps.size())
			writeImEps("imEps."+ossID.str(), imEps);
		//Increment stepID:
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
	//--- time evolution parameters
	const double dt = inputMap.get("dt") * fs; //time interval between reports
	const double tStop = inputMap.get("tStop") * fs; //stopping time for simulation
	const double tStep = inputMap.get("tStep", 0.) * fs; //if non-zero, time step for fixed-step (non-adaptive) integrator
	const double tolAdaptive = inputMap.get("tolAdaptive", 1e-3); //relative tolerance for adaptive integrator
	//--- pump
	const string pumpMode = inputMap.getString("pumpMode"); //must be Evolve, Perturb or Bfield
	if(pumpMode!="Evolve" and pumpMode!="Perturb" and pumpMode!="Bfield")
		die("\npumpMode must be 'Evolve' or 'Perturb' pr 'Bfield'\n");
	const double Tesla = Joule/(Ampere*meter*meter);
	const vector3<> pumpB = inputMap.getVector("pumpB", vector3<>()) * Tesla; //perturbing initial magnetic field in Tesla (used only in Bfield mode)
	const double pumpOmega = inputMap.get("pumpOmega", pumpMode=="Bfield" ? 0. : NAN) * eV; //pump frequency in eV (used only in Evolve or Perturb modes)
	const double pumpA0 = inputMap.get("pumpA0", pumpMode=="Bfield" ? 0. : NAN); //pump pulse amplitude / intensity (Units TBD)
	const double pumpTau = inputMap.get("pumpTau", pumpMode=="Bfield" ? 0. : NAN)*fs; //Gaussian pump pulse width (sigma of amplitude) in fs
	const vector3<complex> pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes
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
	const double omegaMin = inputMap.get("omegaMin", pol.size() ? NAN : 0.) * eV; //start of frequency grid for probe response
	const double omegaMax = inputMap.get("omegaMax", pol.size() ? NAN : 0.) * eV; //end of frequency grid for probe response
	const double domega = inputMap.get("domega", pol.size() ? NAN : 0.) * eV; //frequency resolution for probe calculation
	const double tau = inputMap.get("tau", pol.size() ? NAN : 0.) * fs; //Gaussian probe pulse width (sigma of amplitude) in fs
	//--- general options
	const double dE = inputMap.get("dE") * eV; //energy resolution for distribution functions
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	if(ePhMode!="Off" and ePhMode!="DiagK")
		die("\nePhMode must be 'Off' or 'DiagK'\n");
	const bool ePhEnabled = (ePhMode != "Off");
	const double defectFraction = inputMap.get("defectFraction", 0.); //fractional concentration of defects if any
	const string verboseMode = inputMap.has("verbose") ? inputMap.getString("verbose") : "no"; //must be yes or no
	if(verboseMode!="yes" and verboseMode!="no")
		die("\nverboseMode must be 'yes' or 'no'\n");
	const bool verbose = (verboseMode=="yes");
	const string inFile = inputMap.has("inFile") ? inputMap.getString("inFile") : "ldbd.dat"; //input file name
	const string checkpointFile = inputMap.has("checkpointFile") ? inputMap.getString("checkpointFile") : ""; //checkpoint file name
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("T = %lg\n", T);
	logPrintf("dt = %lg\n", dt);
	logPrintf("tStop = %lg\n", tStop);
	logPrintf("tStep = %lg\n", tStep);
	logPrintf("tolAdaptive = %lg\n", tolAdaptive);
	
	logPrintf("pumpMode = %s\n", pumpMode.c_str());
	if(pumpMode == "Bfield")
	{	logPrintf("pumpB = "); pumpB.print(globalLog, " %lg ");
	}
	else
	{	logPrintf("pumpOmega = %lg\n", pumpOmega);
		logPrintf("pumpA0 = %lg\n", pumpA0);
		logPrintf("pumpTau = %lg\n", pumpTau);
		logPrintf("pumpPol = "); print(globalLog, pumpPol);
	}
	if(pol.size())
	{	for(int iPol=0; iPol<int(pol.size()); iPol++)
		{	logPrintf("pol%d = ", iPol+1);
			print(globalLog, pol[iPol]);
		}
		logPrintf("omegaMin = %lg\n", omegaMin);
		logPrintf("omegaMax = %lg\n", omegaMax);
		logPrintf("domega = %lg\n", domega);
		logPrintf("tau = %lg\n", tau);
	}
	logPrintf("dE = %lg\n", dE);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("defectFraction = %lg\n", defectFraction);
	logPrintf("verbose = %s\n", verboseMode.c_str());
	logPrintf("inFile = %s\n", inFile.c_str());
	logPrintf("checkpointFile = %s\n", checkpointFile.c_str());
	logPrintf("\n");
	
	//Create and initialize lindblad calculator:
	Lindblad lb(dmu, T,
		pumpOmega, pumpA0, pumpTau, pumpPol, (pumpMode=="Evolve"), (pumpMode=="Bfield"), pumpB,
		omegaMin, omegaMax, domega, tau, pol, dE,
		ePhEnabled, defectFraction, verbose, checkpointFile);
	lb.initialize(inFile);
	logPrintf("Initialization completed successfully at t[s]: %9.2lf\n\n", clock_sec());
	logFlush();
	
	logPrintf("%lu active k-points parallelized over %d processes.\n", lb.nk, mpiWorld->nProcesses());
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	if(pumpMode!="Evolve" and (not ePhEnabled))
	{	//Simple probe-pump-probe with no relaxation:
		lb.report(-dt, lb.rho);
		lb.applyPump(); //takes care of optical pump or B-field excitation
		lb.report(0., lb.rho);
	}
	else
	{	double tStart = 0.;
		bool checkpointExists = false;
		if(mpiWorld->isHead())
			checkpointExists = (checkpointFile.length()>0) and (fileSize(checkpointFile.c_str())>0);
		mpiWorld->bcast(checkpointExists);
		if(checkpointExists)
		{	logPrintf("Reading checkpoint from '%s' ... ", checkpointFile.c_str()); logFlush(); 
			//Determine offset of current process data and total expected file length:
			size_t offset = sizeof(int)+sizeof(double); //offset due to header
			for(int jProc=0; jProc<mpiWorld->iProcess(); jProc++)
				offset += sizeof(double) * lb.rhoSize[jProc]; //offset due to data from previous processes
			size_t fsizeExpected = offset;
			for(int jProc=mpiWorld->iProcess(); jProc<mpiWorld->nProcesses(); jProc++)
				fsizeExpected += sizeof(double) * lb.rhoSize[jProc];
			mpiWorld->bcast(fsizeExpected);
			//Open check point file and rrad time header:
			MPIUtil::File fp; mpiWorld->fopenRead(fp, checkpointFile.c_str(), fsizeExpected);
			mpiWorld->fread(&(lb.stepID), sizeof(int), 1, fp);
			mpiWorld->fread(&tStart, sizeof(double), 1, fp);
			mpiWorld->bcast(tStart);
			//Read density matrix from check point file:
			mpiWorld->fseek(fp, offset, SEEK_SET);
			mpiWorld->fread(lb.rho.data(), sizeof(double), lb.rho.size(), fp);
			mpiWorld->fclose(fp);
			logPrintf("done.\n");
		}
		else if(pumpMode!="Evolve")
		{	//Do an initial report akin to above and apply the pump:
			lb.report(-dt, lb.rho);
			lb.applyPump(); //takes care of optical pump or B-field excitation
			tStart = 0.; //integrate will report at t=0 below, before evolving ePh relaxation
		}
		else
		{	//Set start time to a multiple of dt that covers pulse:
			tStart = -dt * ceil(5.*tau/dt);
		}
		//Evolve:
		if(tStep) //Fixed-step integrator:
			lb.integrateFixed(lb.rho, tStart, tStop, tStep, dt);
		else //Adaptive integrator:
			lb.integrateAdaptive(lb.rho, tStart, tStop, tolAdaptive, dt);
	}
	
	//Cleanup:
	FeynWann::finalize();
	return 0;
}
