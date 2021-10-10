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
#include "BlockCyclicMatrix.h"
#include <core/Units.h>

#ifdef PETSC_ENABLED
	#include <petsc.h>
	//Slightly more graceful wrapper to CHKERRQ() macro from Petsc:
	PetscInt iErr = 0;
	#define CHECKERR(codeLine) \
		iErr = codeLine; \
		CHKERRQ(iErr);
#else
	#define PetscErrorCode int
	#define CHECKERR(codeLine) codeLine;
#endif

inline matrix dot(const matrix* P, vector3<complex> pol)
{	return pol[0]*P[0] + pol[1]*P[1] + pol[2]*P[2];
}

//Construct identity - X:
inline diagMatrix bar(const diagMatrix& X)
{	diagMatrix Xbar(X);
	for(double& x: Xbar) x = 1. - x;
	return Xbar;
}

enum ValleyMode
{	ValleyNone,
	ValleyInter,
	ValleyIntra
};

//Lindblad initialization, time evolution and measurement operators using FeynWann callback
struct LindbladLinear
#ifdef PETSC_ENABLED
: public Integrator<DM1>
#endif
{	
	int stepID; //current time and reporting step number
	
	const double dmu, T, invT; //!< Fermi level position relative to neutral value / VBM, and temperature
	const bool spectrumMode; //!< ScaLAPACK diagonalization if yes, linearized real-time dynamics using PETSc otherwise
	const int blockSize; //!< block size in ScaLAPACK matrix distribution
	const double pumpOmega, pumpA0, pumpTau; const vector3<complex> pumpPol; //!< pump parameters
	const bool pumpBfield; const vector3<> pumpB; //pump parameters for Bfield mode
	const double omegaMin, domega, omegaMax; const int nomega; //!< probe frequency grid
	const double tau; const std::vector<vector3<complex>> pol; //!< probe parameters
	const double dE; //!< energy resolution for distribution functions
	
	const bool ePhEnabled; //!< whether e-ph coupling is enabled
	bool detailBalance;
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
	
	ValleyMode valleyMode; //!< whether all k-pairs (None) or only those corresponding to Intra/Inter-valley scattering are included
	const vector3<> K, Kp; //!< K and K' valley in reciprocal lattice coordinates
	static inline vector3<> wrap(const vector3<>& x) { vector3<> result = x;  for(int dir=0; dir<3; dir++) result[dir] -= floor(0.5 + result[dir]); return result; }
	inline bool isKvalley(vector3<> k) const { return (wrap(K-k)).length_squared() < (wrap(Kp-k)).length_squared(); }
	
	LindbladLinear(double dmu, double T, bool spectrumMode, int blockSize,
		double pumpOmega, double pumpA0, double pumpTau, vector3<complex> pumpPol, bool pumpBfield, vector3<> pumpB,
		double omegaMin, double omegaMax, double domega, double tau, std::vector<vector3<complex>> pol, double dE,
		bool ePhEnabled, double defectFraction, bool verbose, string checkpointFile, ValleyMode valleyMode)
	: stepID(0),
		dmu(dmu), T(T), invT(1./T), spectrumMode(spectrumMode), blockSize(blockSize),
		pumpOmega(pumpOmega), pumpA0(pumpA0), pumpTau(pumpTau), pumpPol(pumpPol), pumpBfield(pumpBfield), pumpB(pumpB),
		omegaMin(omegaMin), domega(domega), omegaMax(omegaMax), nomega(1+int(round((omegaMax-omegaMin)/domega))),
		tau(tau), pol(pol), dE(dE), ePhEnabled(ePhEnabled), defectFraction(defectFraction), 
		verbose(verbose), checkpointFile(checkpointFile),
		Emin(+DBL_MAX), Emax(-DBL_MAX),
		valleyMode(valleyMode), K(1./3, 1./3, 0), Kp(-1./3, -1./3, 0)
	{
	}
	
	//---- Flat density matrix storage and access functions ----
	DM1 drho; //!< flat array of density matrix changes of all k stored on this process
	std::vector<size_t> nInnerPrev; //cumulative nInner for each k, which is the offset into the Eall array for each k
	std::vector<size_t> nRhoPrev; //cumulative nInner^2 for each k, which is the offset into the global rho structure for each k
	std::vector<size_t> rhoOffset; //!< array of offsets into process's rho for each k (essentially nRhoPrev - nRhoPrev[ikStart])
	std::vector<size_t> rhoSize; //!< total size of rho on each process
	size_t rhoOffsetGlobal; //!< offset of current process rho data in the overall data
	size_t rhoSizeTot; //!< total size of rho
	
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
	
	//--------- PETSc for linearized time evolution sparse matrix and conversion -----------
	#ifdef PETSC_ENABLED
	Mat evolveMat; //Time evolution operator
	Vec vRho, vRhoDot; //!< temporary copies of drho and rdhoDot data in Petsc format
	
	//Clean up Petsc quantities
	PetscErrorCode cleanup()
	{	if(not spectrumMode)
		{	CHECKERR(MatDestroy(&evolveMat));
			CHECKERR(VecDestroy(&vRho));
			CHECKERR(VecDestroy(&vRhoDot));
		}
		return 0;
	}
	#endif
	
	//--------- Blacs / ScaLAPACK interface for dense diagonalization -----------
	#ifdef SCALAPACK_ENABLED
	std::shared_ptr<BlockCyclicMatrix> bcm;
	BlockCyclicMatrix::Buffer evolveMatDense, spinMatDense, spinPertDense;
	#endif
	
	//--------- Initialize -------------
	PetscErrorCode initialize(string inFile)
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
		std::vector<int> isKall(valleyMode==ValleyNone ? 0 : nk, 0); //whether each k-point is closer to K or K'
		
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
			if(valleyMode != ValleyNone) isKall[ikStart+ikMine] = isKvalley(s.k);
			
			//Set initial occupations:
			s.rho0.resize(s.nInner);
			for(int b=0; b<s.nInner; b++)
				s.rho0[b] = fermi((s.E[b+s.innerStart]-dmu)*invT);
		}
		mpiWorld->fclose(fp);
		if(valleyMode != ValleyNone) mpiWorld->allReduceData(isKall, MPIUtil::ReduceMax);
			
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
		rhoSizeTot = 0;
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t jkStart = kDivision.start(jProc);
			size_t jkStop = kDivision.stop(jProc);
			size_t offset = 0; //start at 0 for each process's chunk
			for(size_t jk=jkStart; jk<jkStop; jk++)
			{	rhoOffset[jk] = offset;
				offset += nInnerAll[jk]*nInnerAll[jk];
			}
			rhoSize[jProc] = offset;
			if(jProc == mpiWorld->iProcess()) rhoOffsetGlobal = rhoSizeTot;
			rhoSizeTot += offset; //cumulative over all processes
		}
		
		//Initialize rho:
		drho.assign(rhoSize[mpiWorld->iProcess()], 0.);
		
		//Initialize sparse matrix corresponding to net time evolution (if required):
		if(ePhEnabled)
		{	//Make inner-window energies available for all processes:
			nInnerPrev.assign(nk+1, 0); //cumulative nInner for each k (offset into the Eall array)
			nRhoPrev.assign(nk+1, 0); //cumulative nInner^2 for each k (offset into global rho)
			for(size_t ik=0; ik<nk; ik++)
			{	nInnerPrev[ik+1] = nInnerPrev[ik] + nInnerAll[ik];
				nRhoPrev[ik+1] = nRhoPrev[ik] +  nInnerAll[ik]*nInnerAll[ik];
			}
			std::vector<double> Eall(nInnerPrev.back()); //inner window energies for all k
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
			//Collect matrix elements in triplet format:
			#ifdef PETSC_ENABLED
			std::vector<int> nnzD, nnzO; //number of process-diagonal and process off-diagonal entries by row
			if(not spectrumMode) { nnzD.resize(rhoSizeTot); nnzO.resize(rhoSizeTot); }
			#endif
			#ifdef SCALAPACK_ENABLED
			std::vector<std::vector<std::pair<double,int>>> evolveEntries(mpiWorld->nProcesses()); //list of entries by destination process
			if(spectrumMode) bcm = std::make_shared<BlockCyclicMatrix>(rhoSizeTot, blockSize, mpiWorld); //ScaLAPACK wrapper object
			#endif
			int nPasses = spectrumMode ? 1 : 2; //run two passes for PETSc: first for size determination, second to set the entries
			logPrintf(spectrumMode
				? "Initializing time evolution operator ... "
				: "Determining sparsity structure of time evolution operator ... "); logFlush();
			for(int iPass=0; iPass<nPasses; iPass++)
			{	State* sPtr = state.data();
				for(size_t ik1=ikStart; ik1<ikStop; ik1++)
				{	State& s = *(sPtr++);
					const double* E1 = &(s.E[s.innerStart]);
					const int& nRhoPrev1 = nRhoPrev[ik1];
					const int& nInner1 = nInnerAll[ik1];
					const int N1 = nInner1*nInner1; //number of density matrix entries
					const int whose1 = mpiWorld->iProcess();
					const diagMatrix& f1 = s.rho0;
					const diagMatrix f1bar = bar(f1);
					#ifdef SCALAPACK_ENABLED
					//Coherent evolution (only in spectrum mode):
					if(spectrumMode)
					{	for(int a=0; a<nInner1; a++)
							for(int b=0; b<nInner1; b++)
								if(a != b)
								{	int iRow = nRhoPrev1+a+b*nInner1;
									int iCol = nRhoPrev1+b+a*nInner1;
									double Ediff = E1[b]-E1[a];
									int localIndex, whose = bcm->globalIndex(iRow, iCol, localIndex);
									evolveEntries[whose].push_back(std::make_pair(Ediff,localIndex));
								}
					}
					#endif
					//Electron-phonon part:
					const double prefacEph = 2*M_PI/nkTot; //factor of 2 from the +h.c. contribution
					std::vector<LindbladFile::GePhEntry>::iterator g = s.GePh.begin();
					while(g != s.GePh.end())
					{	const size_t& ik2 = g->jk;
						const int& nInner2 = nInnerAll[ik2];
						const int N2 = nInner2*nInner2; //number of density matrix entries
						const int& nRhoPrev2 = nRhoPrev[ik2];
						const int whose2 = whose(ik2);
						const double* E2 = &(Eall[nInnerPrev[ik2]]);
						diagMatrix f2(nInner2); for(int b2=0; b2<nInner2; b2++) f2[b2] = fermi(invT*(E2[b2]-dmu));
						const diagMatrix f2bar = bar(f2);
						//Skip combinations if necessary based on valleyMode:
						bool shouldSkip = false; //must skip after iterating over all matching k2 below (else endless loop!)
						if((valleyMode==ValleyIntra) and (isKall[ik1]!=isKall[ik2])) shouldSkip=true; //skip intervalley scattering
						if((valleyMode==ValleyInter) and (isKall[ik1]==isKall[ik2])) shouldSkip=true; //skip intravalley scattering
						//Store results in dense complex blocks of the superoperator first:
						matrix L12 = zeroes(N1,N2); complex* L12data = L12.data();
						matrix L21 = zeroes(N2,N1); complex* L21data = L21.data();
						matrix L11 = zeroes(N1,N1); complex* L11data = L11.data();
						matrix L22 = zeroes(N2,N2); complex* L22data = L22.data();
						#define L(i,a,b, j,c,d) L##i##j##data[L##i##j.index(a+b*nInner##i, c+d*nInner##j)] //access superoperator block element
						//Loop over all connections to the same ik2:
						while((g != s.GePh.end()) and (g->jk == ik2))
						{	g->G.init(nInner1, nInner2);
							if (!detailBalance) //JX
								g->initA(T, defectFraction);
							else
								g->initA(E1, E2, T, defectFraction);
							//Loop over A- and A+
							for(int pm=0; pm<2; pm++) 
							{	const SparseMatrix& Acur = pm ? g->Ap : g->Am;
								const diagMatrix& f1cur = pm ? f1 : f1bar;
								const diagMatrix& f2cur = pm ? f2bar : f2;
								//Loop oover all pairs of non-zero entries:
								for(const SparseEntry& s1: Acur)
								{	int a = s1.i, b = s1.j; //to match derivation's notation
									for(const SparseEntry& s2: Acur)
									{	int c = s2.i, d = s2.j; //to match derivation's notation
										complex M = prefacEph * (s1.val * s2.val.conj());
										L(1,a,c, 2,b,d) += f1cur[a] * M;
										L(2,d,b, 1,c,a) += f2cur[d] * M;
										if(b == d) for(int e=0; e<nInner1; e++) L(1,e,c, 1,e,a) -= f2cur[b] * M;
										if(a == c) for(int e=0; e<nInner2; e++) L(2,e,b, 2,e,d) -= f1cur[c] * M;
									}
								}
							}
							//Move to next element:
							g++;
						}
						if(shouldSkip) continue;
						#undef L
						//Convert from complex to real input and real outputs (based on h.c. symmetry):
						#define CreateRandInv(i) \
							SparseMatrix R##i(N##i,N##i,2*N##i), Rinv##i(N##i,N##i,2*N##i); \
							for(int a=0; a<nInner##i; a++) \
							{	for(int b=0; b<a; b++) \
								{	int ab = a+b*nInner##i, ba = b+a*nInner##i; \
									R##i.push_back(SparseEntry{ab,ab,complex(1,0)}); R##i.push_back(SparseEntry{ab,ba,complex(0,+1)}); \
									R##i.push_back(SparseEntry{ba,ab,complex(1,0)}); R##i.push_back(SparseEntry{ba,ba,complex(0,-1)}); \
									Rinv##i.push_back(SparseEntry{ab,ab,complex(+0.5,0)}); Rinv##i.push_back(SparseEntry{ab,ba,complex(+0.5,0)}); \
									Rinv##i.push_back(SparseEntry{ba,ab,complex(0,-0.5)}); Rinv##i.push_back(SparseEntry{ba,ba,complex(0,+0.5)}); \
								} \
								int aa = a+a*nInner##i; \
								R##i.push_back(SparseEntry{aa,aa,1.}); \
								Rinv##i.push_back(SparseEntry{aa,aa,1.}); \
							}
						CreateRandInv(1)
						CreateRandInv(2)
						#undef CreateRandInv
						L12 = Rinv1 * (L12 * R2);
						L21 = Rinv2 * (L21 * R1);
						L11 = Rinv1 * (L11 * R1);
						L22 = Rinv2 * (L22 * R2);
						//Extract / count / set non-zero entries depending on the mode and pass:
						if(spectrumMode)
						{
							#ifdef SCALAPACK_ENABLED
							#define EXTRACT_NNZ(i,j) \
							{	const complex* data = L##i##j.data(); \
								for(int col=0; col<L##i##j.nCols(); col++) \
								{	for(int row=0; row<L##i##j.nRows(); row++) \
									{	double M = (data++)->real(); \
										if(M) \
										{	int iRow = row+nRhoPrev##i; \
											int iCol = col+nRhoPrev##j; \
											int localIndex, whose = bcm->globalIndex(iRow, iCol, localIndex); \
											evolveEntries[whose].push_back(std::make_pair(M,localIndex)); \
										} \
									} \
								} /* TODO */ \
							}
							EXTRACT_NNZ(1,2)
							EXTRACT_NNZ(2,1)
							EXTRACT_NNZ(1,1)
							EXTRACT_NNZ(2,2)
							#undef EXTRACT_NNZ
							#endif
						}
						else
						{
							#ifdef PETSC_ENABLED
							if(iPass == 0)
							{	//Count non-zero matrix elements:
								#define COUNT_NNZ(i,j) \
								{	std::vector<int>& nnz = (whose##i == whose##j) ? nnzD : nnzO; \
									const complex* data = L##i##j.data(); \
									for(int col=0; col<L##i##j.nCols(); col++) \
									{	for(int row=0; row<L##i##j.nRows(); row++) \
										{	double M = (data++)->real(); \
											if(M) nnz[row+nRhoPrev##i]++; \
										} \
									} \
								}
								COUNT_NNZ(1,2)
								COUNT_NNZ(2,1)
								COUNT_NNZ(1,1)
								COUNT_NNZ(2,2)
								#undef COUNT_NNZ
							}
							else
							{	//Set matrix elements in PETSc matrix:
								#define SET_NNZ(i,j) \
								{	const complex* data = L##i##j.data(); \
									for(int col=0; col<L##i##j.nCols(); col++) \
									{	for(int row=0; row<L##i##j.nRows(); row++) \
										{	double M = (data++)->real(); \
											if(M) CHECKERR(MatSetValue(evolveMat, row+nRhoPrev##i, col+nRhoPrev##j, M, ADD_VALUES)); \
										} \
									} \
								}
								SET_NNZ(1,2)
								SET_NNZ(2,1)
								SET_NNZ(1,1)
								SET_NNZ(2,2)
								#undef SET_NNZ
							}
							#endif
						}
					}
					if(iPass+1 == nPasses) s.GePh.clear(); //no longer needed; optimize memory
				}
				if(spectrumMode) logPrintf("done.\n"); 
				#ifdef PETSC_ENABLED
				else
				{	if(iPass == 0)
					{	//Allocate matrix knowing sparsity structure at the end of the first pass
						int N = rhoSizeTot;
						int Nmine = rhoSize[mpiWorld->iProcess()];
						MPI_Allreduce(MPI_IN_PLACE, nnzD.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
						MPI_Allreduce(MPI_IN_PLACE, nnzO.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
						for(size_t i=rhoOffsetGlobal; i<rhoOffsetGlobal+Nmine; i++)
						{	nnzD[i] = std::min(nnzD[i], Nmine);
							nnzO[i] = std::min(nnzO[i], N - Nmine);
						}
						logPrintf("done.\nInitializing PETSc sparse matrix for time evolution ... "); logFlush();
						CHECKERR(MatCreateAIJ(PETSC_COMM_WORLD, Nmine, Nmine, N, N,
							0, nnzD.data()+rhoOffsetGlobal, 0, nnzO.data()+rhoOffsetGlobal, &evolveMat));
						CHECKERR(MatSetOption(evolveMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
					}
					else
					{	//Finalize matrix assembly at end of second pass:
						CHECKERR(MatAssemblyBegin(evolveMat, MAT_FINAL_ASSEMBLY));
						CHECKERR(MatAssemblyEnd(evolveMat, MAT_FINAL_ASSEMBLY));
						MatInfo info; CHECKERR(MatGetInfo(evolveMat, MAT_GLOBAL_SUM, &info));
						logPrintf("done. Net sparsity: %.0lf non-zero in %lu x %lu matrix (%.1lf%% fill)\n",
							info.nz_used, rhoSizeTot, rhoSizeTot, info.nz_used*100./(rhoSizeTot*rhoSizeTot));
						logFlush();
						CHECKERR(MatCreateVecs(evolveMat, &vRho, &vRhoDot));
					}
				}
				#endif
			}
			
			//Convert from triplet to appropriate format:
			if(spectrumMode)
			{	//Convert to dense matrix for ScaLAPACK:
				#ifdef SCALAPACK_ENABLED
				logPrintf("Converting to block-cyclic distributed dense matrix ... "); logFlush();
				//--- sync sizes of remote pieces:
				std::vector<size_t> nEntriesFromProc(mpiWorld->nProcesses());
				{	std::vector<size_t> nEntriesToProc(mpiWorld->nProcesses());
					std::vector<MPIUtil::Request> requests(2*(mpiWorld->nProcesses()-1));
					int iRequest = 0;
					for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
						if(jProc != mpiWorld->iProcess())
						{	nEntriesToProc[jProc] = evolveEntries[jProc].size();
							mpiWorld->send(&nEntriesToProc[jProc], 1, jProc, 0, &requests[iRequest++]);
							mpiWorld->recv(&nEntriesFromProc[jProc], 1, jProc, 0, &requests[iRequest++]);
							
						}
					mpiWorld->waitAll(requests);
				}
				//--- transfer remote pieces:
				std::vector<std::vector<std::pair<double,int>>> evolveEntriesMine(mpiWorld->nProcesses());
				{	std::vector<MPIUtil::Request> requests(2*(mpiWorld->nProcesses()-1));
					int iRequest = 0;
					for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
						if(jProc == mpiWorld->iProcess())
							std::swap(evolveEntries[jProc], evolveEntriesMine[jProc]);
						else
						{	evolveEntriesMine[jProc].resize(nEntriesFromProc[jProc]);
							MPI_Irecv(evolveEntriesMine[jProc].data(), evolveEntriesMine[jProc].size(), MPI_DOUBLE_INT, jProc, 1, MPI_COMM_WORLD, &requests[iRequest++]);
							MPI_Isend(evolveEntries[jProc].data(), evolveEntries[jProc].size(), MPI_DOUBLE_INT, jProc, 1, MPI_COMM_WORLD, &requests[iRequest++]);
						}
					mpiWorld->waitAll(requests);
					evolveEntries.clear();
				}
				//--- set to dense matrix:
				size_t nNZ = 0;
				evolveMatDense.assign(bcm->nDataMine, 0.);
				for(const auto& entries: evolveEntriesMine)
				{	for(const std::pair<double,int>& entry: entries)
						evolveMatDense[entry.second] += entry.first;
					nNZ += entries.size();
				}
				mpiWorld->allReduce(nNZ, MPIUtil::ReduceSum);
				logPrintf("done. Total terms: %lu in %lu x %lu matrix (%.1lf%% fill)\n",
					nNZ, rhoSizeTot, rhoSizeTot, nNZ*100./(rhoSizeTot*rhoSizeTot));
				logFlush();
				
				//Compute spin and magnetic field vectors:
				double Bmag = pumpB.length(); //perturbation strength set by input, but all components calculated
				if(not Bmag)
				{	const double Tesla = Joule/(Ampere*meter*meter);
					Bmag = 1.*Tesla;
					logPrintf("Setting test |B| = 1 Tesla for B-field perturbation matrix. (Use pumpB to override if needed.)\n");
				}
				logPrintf("Initializing spin matrix elements ... "); logFlush();
				size_t rhoSizeMine = drho.size();
				DM1 spinMat(rhoSizeMine*6); //6 columns: Sx Sy Sz dRho(Bx) dRho(By) dRho(Bz)
				const State* sPtr = state.data();
				const double prefac = spinWeight*(1./nkTot); //BZ integration weight
				for(size_t ik=ikStart; ik<ikStop; ik++)
				{	const State& s = *(sPtr++);
					for(int iDir=0; iDir<3; iDir++)
					{	//Spin matrix element to column iDir:
						matrix OSmat = s.S[iDir] - 0.5*diag(s.S[iDir]); //S with an overlap factor such that S^rho = Tr[rho*S]
						accumRhoHC(prefac*OSmat, spinMat.data()+(rhoOffset[ik]+iDir*rhoSizeMine));
						//Magnetic field perturbation to column 3+iDir:
						matrix Htot(s.E(s.innerStart, s.innerStart+s.nInner));
						Htot -= Bmag * s.S[iDir];
						//--- compute Fermi function perturbation:
						diagMatrix Epert; matrix Vpert;
						Htot.diagonalize(Vpert, Epert);
						diagMatrix fPert(s.nInner);
						for(int b=0; b<s.nInner; b++)
							fPert[b] = fermi((Epert[b]-dmu)*invT);
						matrix rhoPert = Vpert * fPert * dagger(Vpert);
						accumRhoHC(0.5*(rhoPert-s.rho0), spinMat.data()+(rhoOffset[ik]+(iDir+3)*rhoSizeMine));
					}
				}
				
				//Redistribute to match ScaLAPACK matrices:
				spinMatDense.resize(bcm->nRowsMine*3); //spin matrix
				spinPertDense.resize(bcm->nRowsMine*3); //B-field perturbation
				int jProc = mpiWorld->iProcess();
				int iProcPrev = positiveRemainder(mpiWorld->iProcess()-1, mpiWorld->nProcesses());
				int iProcNext = positiveRemainder(mpiWorld->iProcess()+1, mpiWorld->nProcesses());
				for(int iProcShift=0; iProcShift<mpiWorld->nProcesses(); iProcShift++)
				{	//Set local matrix elements from spinMat to spinMatDense:
					int iRowStart = nRhoPrev[kDivision.start(jProc)]; //global start row of current data block
					int iRowStop = nRhoPrev[kDivision.stop(jProc)]; //global stop row of current data block
					int nRowsCur = rhoSize[jProc];
					assert(iRowStop - iRowStart == nRowsCur);
					int iRowMineStart, iRowMineStop; //local row indices that match
					bcm->getRange(bcm->iRowsMine, iRowStart, iRowStop, iRowMineStart, iRowMineStop);
					for(int iRowMine=iRowMineStart; iRowMine<iRowMineStop; iRowMine++)
					{	int iRow = bcm->iRowsMine[iRowMine];
						for(int iCol=0; iCol<3; iCol++)
						{	spinMatDense[iRowMine+iCol*bcm->nRowsMine] = spinMat[(iRow-iRowStart)+iCol*nRowsCur];
							spinPertDense[iRowMine+iCol*bcm->nRowsMine] = spinMat[(iRow-iRowStart)+(iCol+3)*nRowsCur];
						}
					}
					//Circulate spinMat in communication ring:
					if((iProcShift+1) == mpiWorld->nProcesses()) break;
					int jProcNext = (jProc + 1) % mpiWorld->nProcesses();
					DM1 spinMatNext(rhoSize[jProcNext]*6);
					std::vector<MPIUtil::Request> request(2);
					mpiWorld->sendData(spinMat, iProcPrev, iProcShift, &request[0]);
					mpiWorld->recvData(spinMatNext, iProcNext, iProcShift, &request[1]);
					mpiWorld->waitAll(request);
					std::swap(spinMat, spinMatNext);
					jProc = jProcNext;
				}
				logPrintf("done.\n");
				#endif
			}
		}
		logPrintf("\n"); logFlush();
		return 0;
	}
	
#ifdef PETSC_ENABLED
	
	//Calculate change in probe response due to current drho:
	diagMatrix calcDeltaImEps(double t, const DM1& drho) const
	{	static StopWatch watch("Lindblad::calcImEps");
		size_t nImEps = pol.size() * nomega;
		if(nImEps==0) return diagMatrix(); //no probe specified
		watch.start();
		diagMatrix dimEps(nImEps);
		//Collect contributions from each k at this process:
		const State* sPtr = state.data();
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			const matrix drhoCurSub = getRho(drho.data()+rhoOffset[ik], s.nInner);
			//Expand density matrix:
			matrix drhoCur = zeroes(s.nOuter, s.nOuter);
			drhoCur.set(s.innerStart,s.innerStop, s.innerStart,s.innerStop, drhoCurSub);
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
					//Compute change in rho due to probe (summed over excitation/deexcitational already):
					diagMatrix deltaRhoDiag = diag(Pdag*drhoCur*P + P*drhoCur*Pdag - drhoCur*P*Pdag - Pdag*P*drhoCur);
					dimEps[iPol*nomega+iomega] += prefac * dot(s.E, deltaRhoDiag);
				}
			}
		}
		//Accumulate contributions from all processes on head:
		mpiWorld->reduceData(dimEps, MPIUtil::ReduceSum);
		watch.stop();
		return dimEps;
	}
	
	//Write change in imEps to plain-text file:
	void writeDeltaImEps(string fname, const diagMatrix& dimEps) const
	{	if(mpiWorld->isHead())
		{	ofstream ofs(fname);
			ofs << "#omega[eV]";
			for(int iPol=0; iPol<int(pol.size()); iPol++)
				ofs << " dImEps" << (iPol+1);
			ofs << "\n";
			for(int iomega=0; iomega<nomega; iomega++)
			{	double omega = omegaMin + iomega*domega;
				ofs << omega/eV;
				for(int iPol=0; iPol<int(pol.size()); iPol++)
					ofs << '\t' << dimEps[iPol*nomega+iomega];
				ofs << '\n';
			}
		}
	}
	
	//Apply pump using perturbation theory (instantly go from before to after pump, skipping time evolution)
	void applyPump()
	{	static StopWatch watch("Lindblad::applyPump"); 
		watch.start();
		const State* sPtr = state.data();
		//Perturb each k separately:
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
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
				accumRhoHC(0.5*(rhoPert-s.rho0), drho.data()+rhoOffset[ik]);
			}
			else
			{	const diagMatrix& rho0 = s.rho0;
				matrix rho0bar(bar(rho0)); //1-rho0
				//Compute and apply perturbation:
				matrix P = s.pumpPD; //P-
				matrix Pdag = dagger(P); //P+
				matrix deltaRho;
				for(int s=-1; s<=+1; s+=2)
				{	deltaRho += rho0bar*P*rho0*Pdag - Pdag*rho0bar*P*rho0;
					std::swap(P, Pdag); //P- <--> P+
				}
				accumRhoHC((M_PI*pumpA0*pumpA0) * deltaRho, drho.data()+rhoOffset[ik]);
			}
		}
		watch.stop();
	}
	
	//Time evolution operator returning drho/dt
	DM1 compute(double t, const DM1& drho)
	{	static StopWatch watchEph("Lindblad::compute::ePh");
		
		DM1 drhoDot(drho.size(), 0.);
		
		//E-ph relaxation contribution:
		if(ePhEnabled)
		{	watchEph.start();
			//Convert interaction picture rho data to Schrodinger picture version in PETSc format:
			double* vRhoPtr;  VecGetArray(vRho, &vRhoPtr);
			eblas_zero(drho.size(), vRhoPtr);
			std::vector<complex> phase(drho.size());
			complex* phaseData = phase.data();
			const State* sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				const double* Einner = s.E.data() + s.innerStart;
				matrix drhoCur = getRho(drho.data()+rhoOffset[ik], s.nInner);
				complex* drhoData = drhoCur.data();
				for(int bCol=0; bCol<s.nInner; bCol++)
					for(int bRow=0; bRow<s.nInner; bRow++)
					{	complex phase = 0.5*cis(t*(Einner[bCol]-Einner[bRow])); //factor of 1/2 in order to use accumRhoHC
						*(drhoData++) *= phase;
						*(phaseData++) = phase.conj(); //cache the reverse phase for below
					}
				accumRhoHC(drhoCur, vRhoPtr+rhoOffset[ik]);
			}
			VecRestoreArray(vRho, &vRhoPtr);
			//Apply sparse operator using PETSc:
			MatMult(evolveMat, vRho, vRhoDot);
			//Copy Schrodinger picture rhoDot data in PETSc format back to interaction picture:
			const double* vRhoDotPtr;  VecGetArrayRead(vRhoDot, &vRhoDotPtr);
			sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				matrix rhoDotCur = getRho(vRhoDotPtr+rhoOffset[ik], s.nInner);
				eblas_zmul(rhoDotCur.nData(), phase.data()+rhoOffset[ik],1, rhoDotCur.data(),1);
				accumRhoHC(rhoDotCur, drhoDot.data()+rhoOffset[ik]);
			}
			VecRestoreArrayRead(vRhoDot, &vRhoDotPtr);
			watchEph.stop();
		}
		
		if(verbose)
		{	//Report current statistics:
			double drhoDotMax = 0., drhoEigMin = +DBL_MAX, drhoEigMax = -DBL_MAX;
			const State* sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				//max(rhoDot)
				const matrix drhoDotCur = getRho(drhoDot.data()+rhoOffset[ik], s.nInner);
				drhoDotMax = std::max(drhoDotMax, drhoDotCur.data()[cblas_izamax(drhoDotCur.nData(), drhoDotCur.data(), 1)].abs());
				//eig(rho):
				const matrix drhoCur = getRho(drho.data()+rhoOffset[ik], s.nInner);
				matrix V; diagMatrix f;
				drhoCur.diagonalize(V, f);
				drhoEigMin = std::min(drhoEigMin, f.front());
				drhoEigMax = std::max(drhoEigMax, f.back());
			}
			mpiWorld->reduce(drhoDotMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(drhoEigMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(drhoEigMin, MPIUtil::ReduceMin);
			logPrintf("\n\tComputed at t[fs]: %lg  max(drhoDot): %lg drhoEigRange: [ %lg %lg ] ",
				t/fs, drhoDotMax, drhoEigMin, drhoEigMax); logFlush();
		}
		else logPrintf("(t[fs]: %lg) ", t/fs);
		logFlush();
		
		return drhoDot;
	}
	
	//Print / dump quantities at each checkpointed step / eigenmode
	void report(double t, const DM1& drho) const
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
			const matrix drhoCur = getRho(drho.data()+rhoOffset[ik], s.nInner);
			//Energy and distribution:
			const complex* drhoData = drhoCur.data();
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
			{	const complex* drhoData = drhoCur.data();
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
			logPrintf("Integrate: Step: %4d   t[fs]: %6.1lf   Etot[eV]: %.6lf", stepID, t/fs, Etot/eV);
			logPrintf("   dfMax: %6.4lf", dfMax);
			if(spinorial) logPrintf("   S: [ %16.15lg %16.15lg %16.15lg ]", Stot[0],  Stot[1],  Stot[2]);
			logPrintf("\n"); logFlush();
			/*
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
			}*/
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
				fwrite(drho.data(), sizeof(double), drho.size(), fp);
				//Data from remaining processes:
				for(int jProc=1; jProc<mpiWorld->nProcesses(); jProc++)
				{	DM1 buf(rhoSize[jProc]);
					mpiWorld->recvData(buf, jProc, 0); //recv data to be written
					fwrite(buf.data(), sizeof(double), buf.size(), fp);
				}
				fclose(fp);
			}
			else mpiWorld->sendData(drho, 0, 0); //send to head for writing
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
			mpiWorld->fwrite(drho.data(), sizeof(double), drho.size(), fp);
			mpiWorld->fclose(fp);
			#endif
		}
		watch.stop();
		//Probe responses if present:
		diagMatrix imEps = calcDeltaImEps(t, drho);
		if(imEps.size())
			writeDeltaImEps("dimEps."+ossID.str(), imEps);
		//Increment stepID:
		((LindbladLinear*)this)->stepID++;
	}
#endif
};

inline void print(FILE* fp, const vector3<complex>& v, const char* format="%lg ")
{	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].real()); std::fprintf(fp, "] + 1j*");
	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].imag()); std::fprintf(fp, "]\n");
}
inline vector3<complex> normalize(const vector3<complex>& v) { return v * (1./sqrt(v[0].norm() + v[1].norm() + v[2].norm())); }

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Lindblad linearized dynamics or spectrum in an ab initio Wannier basis");
	
	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- doping / temperature
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional: shift in fermi level from neutral value / VBM in eV (default: 0)
	const double T = inputMap.get("T") * Kelvin; //temperature in Kelvin (ambient phonon T = initial electron T)
	const string mode = inputMap.getString("mode"); //RealTime or Spectrum or SpectrumSparse
	if((mode!="RealTime") and (mode!="Spectrum"))
		die("\nmode must be 'RealTime' or 'Spectrum'\n");
	const bool spectrumMode = (mode == "Spectrum");
	#ifndef SCALAPACK_ENABLED
	if(spectrumMode)
		die("\nSpectrum (dense diagonalization) mode requires linking with ScaLAPACK.\n");
	#endif
	#ifndef PETSC_ENABLED
	if(not spectrumMode)
		die("\nRealTime (linearized time evolution) mode requires linking with PETSc.\n");
	#endif
	//--- eiegen-decomposition parameters (required and used only in spectrum mode)
	const int blockSize = int(inputMap.get("blockSize", 64));
	const string diagMethodName = inputMap.has("diagMethod") ? inputMap.getString("diagMethod") : "PDHSEQRm";
	#ifdef SCALAPACK_ENABLED
	BlockCyclicMatrix::DiagMethod diagMethod;
	EnumStringMap<BlockCyclicMatrix::DiagMethod> diagMethodMap(
		BlockCyclicMatrix::UsePDGEEVX, "PDGEEVX",
		BlockCyclicMatrix::UsePDHSEQR, "PDHSEQR",
		BlockCyclicMatrix::UsePDHSEQRm, "PDHSEQRm"
	);
	if(not diagMethodMap.getEnum(diagMethodName.c_str(), diagMethod))
		die("diagMethod must be one of %s\n", diagMethodMap.optionList().c_str());
	#endif
	//--- time evolution parameters (required and used only in real time mode)
	const double dt = inputMap.get("dt", spectrumMode ? 0. : NAN) * fs; //time interval between reports
	const double tStop = inputMap.get("tStop", spectrumMode ? 0. : NAN) * fs; //stopping time for simulation
	const double tStep = inputMap.get("tStep", 0.) * fs; //if non-zero, time step for fixed-step (non-adaptive) integrator
	const double tolAdaptive = inputMap.get("tolAdaptive", 1e-3); //relative tolerance for adaptive integrator
	//--- pump / Bfield (required and used only in real time mode)
	const string pumpMode = spectrumMode ? "Bfield" : inputMap.getString("pumpMode"); //must be Perturb or Bfield (Evolve not allowed)
	if(pumpMode!="Perturb" and pumpMode!="Bfield")
		die("\npumpMode must be 'Perturb' or 'Bfield' (Evolve not yet supported by lindbladLinear)\n");
	const double Tesla = Joule/(Ampere*meter*meter);
	const vector3<> pumpB = inputMap.getVector("pumpB", vector3<>()) * Tesla; //perturbing initial magnetic field in Tesla (used only in Bfield mode)
	const double pumpOmega = inputMap.get("pumpOmega", (spectrumMode or pumpMode=="Bfield") ? 0. : NAN) * eV; //pump frequency in eV (used only in Evolve or Perturb modes)
	const double pumpA0 = inputMap.get("pumpA0", (spectrumMode or pumpMode=="Bfield") ? 0. : NAN); //pump pulse amplitude / intensity (Units TBD)
	const double pumpTau = inputMap.get("pumpTau", (spectrumMode or pumpMode=="Bfield") ? 0. : NAN)*fs; //Gaussian pump pulse width (sigma of amplitude) in fs
	const vector3<complex> pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes (used in both modes; parameters required only if one or more polarizations specified)
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
	const double tau = inputMap.get("tau", pol.size() ? NAN :  0.) * fs; //Gaussian probe pulse width (sigma of amplitude) in fs
	//--- general options
	const double dE = inputMap.get("dE") * eV; //energy resolution for distribution functions
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	if(ePhMode!="Off" and ePhMode!="DiagK")
		die("\nePhMode must be 'Off' or 'DiagK'\n");
	const bool ePhEnabled = (ePhMode != "Off");
	if(spectrumMode and not ePhEnabled)
		die("\nePhMode must be 'DiagK' in Spectrum mode\n");
	const double defectFraction = inputMap.get("defectFraction", 0.); //fractional concentration of defects if any
	const string verboseMode = inputMap.has("verbose") ? inputMap.getString("verbose") : "no"; //must be yes or no
	if(verboseMode!="yes" and verboseMode!="no")
		die("\nverboseMode must be 'yes' or 'no'\n");
	const bool verbose = (verboseMode=="yes");
	const string inFile = inputMap.has("inFile") ? inputMap.getString("inFile") : "ldbd.dat"; //input file name
	const string checkpointFile = (inputMap.has("checkpointFile") and (not spectrumMode)) ? inputMap.getString("checkpointFile") : ""; //checkpoint file name
	const string evecFile = inputMap.has("evecFile") ? inputMap.getString("evecFile") : "ldbd.evecs"; //eigenvector file name
	const string valleyModeStr = inputMap.has("valleyMode") ? inputMap.getString("valleyMode") : "None";
	EnumStringMap<ValleyMode> valleyModeMap(ValleyNone, "None", ValleyInter, "Inter", ValleyIntra, "Intra");
	ValleyMode valleyMode;
	if(not valleyModeMap.getEnum(valleyModeStr.c_str(), valleyMode))
		die("\nvalleyMode must be 'None' or 'Intra' or 'Inter'\n");
	const bool detailBalance = inputMap.get("detailBalance", 0);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("T = %lg\n", T);
	logPrintf("mode = %s\n", mode.c_str());
	if(spectrumMode)
	{	logPrintf("blockSize = %d\n", blockSize);
		logPrintf("diagMethod = %s\n", diagMethodName.c_str());
		logPrintf("pumpB = "); pumpB.print(globalLog, " %lg "); //sets magnitude of perturbation
	}
	else
	{	logPrintf("dt = %lg\n", dt);
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
	if(not spectrumMode) logPrintf("checkpointFile = %s\n", checkpointFile.c_str());
	if(spectrumMode) logPrintf("evecFile = %s\n", evecFile.c_str());
	logPrintf("valleyMode = %s\n", valleyModeMap.getString(valleyMode));
	logPrintf("detailBalance = %d\n", detailBalance);
	logPrintf("\n");
	
	//Initialize PETSc if necessary:
	#ifdef PETSC_ENABLED
	if(not spectrumMode)
	{	int argcSlepc=1;
		CHECKERR(PetscInitialize(&argcSlepc, &argv, (char*)0, "")); //don't let petsc see the actual command line (too many conflicts)
	}
	#endif
	
	//Create and initialize lindblad calculator:
	LindbladLinear lbl(dmu, T, spectrumMode, blockSize,
		pumpOmega, pumpA0, pumpTau, pumpPol, (pumpMode=="Bfield"), pumpB,
		omegaMin, omegaMax, domega, tau, pol, dE,
		ePhEnabled, defectFraction, verbose, checkpointFile, valleyMode);
	lbl.detailBalance = detailBalance;
	CHECKERR(lbl.initialize(inFile));
	logPrintf("Initialization completed successfully at t[s]: %9.2lf\n\n", clock_sec());
	logFlush();
	
	if(not spectrumMode) logPrintf("%lu active k-points parallelized over %d processes.\n", lbl.nk, mpiWorld->nProcesses());
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		#ifdef PETSC_ENABLED
		if(not spectrumMode) CHECKERR(PetscFinalize());
		#endif
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	if(spectrumMode)
	{
		//----------- Dense diagonalization using ScaLAPACK ---------------
		#ifdef SCALAPACK_ENABLED
		BlockCyclicMatrix::Buffer VL, VR, spinPert, spinMat;
		std::vector<complex> evals = lbl.bcm->diagonalize(lbl.evolveMatDense, VR, VL, diagMethod, false); //diagonalize
		lbl.bcm->checkDiagonalization(lbl.evolveMatDense, VR, VL, evals); //check diagonalization
		lbl.bcm->matMultVec(1., VL, lbl.spinPertDense, spinPert); //weight of each eigenmode in each B-field perturbation
		lbl.bcm->matMultVec(1., VR, lbl.spinMatDense, spinMat); //spin matrix elements of each eigenmode
		logPrintf("\n%19s %19s %19s %19s %19s %19s %19s %19s\n", "Re(eig)", "Im(eig)",
			"rho1(Bx)", "rho1(By)", "rho1(Bz)", "Sx", "Sy", "Sz");
		int nBlocks = ceildiv(lbl.bcm->N, blockSize);
		for(int iBlock=0; iBlock<nBlocks; iBlock++)
		{	//Make block of eigenvector overlaps on all processes:
			int whose = iBlock % lbl.bcm->nProcsCol;
			int iEigStart = iBlock*blockSize;
			int iEigStop = std::min((iBlock+1)*blockSize, lbl.bcm->N);
			int blockSizeCur = iEigStop-iEigStart;
			BlockCyclicMatrix::Buffer spinMatBlock(blockSizeCur*3), spinPertBlock(blockSizeCur*3);
			if(whose == lbl.bcm->iProcCol)
			{	int eigStartLocal = (iBlock / lbl.bcm->nProcsCol) * blockSize;
				for(int j=0; j<3; j++)
					for(int i=0; i<blockSizeCur; i++)
					{	spinPertBlock[i+j*blockSizeCur] = spinPert[eigStartLocal+i+j*lbl.bcm->nColsMine];
						spinMatBlock[i+j*blockSizeCur] = spinMat[eigStartLocal+i+j*lbl.bcm->nColsMine];
					}
			}
			lbl.bcm->mpiRow->bcastData(spinPertBlock, whose);
			lbl.bcm->mpiRow->bcastData(spinMatBlock, whose);
			for(int i=0; i<blockSizeCur; i++)
			{	int iEig = iEigStart+i;
				logPrintf("%19.12le %19.12le %19.12le %19.12le %19.12le %19.12le %19.12le %19.12le\n",
					evals[iEig].real(), evals[iEig].imag(),
					spinPertBlock[i], spinPertBlock[i+blockSizeCur], spinPertBlock[i+2*blockSizeCur],
					spinMatBlock[i], spinMatBlock[i+blockSizeCur], spinMatBlock[i+2*blockSizeCur] );
			}
		}
		logPrintf("\n");
		//Eigenvector output:
		if(evecFile != "None")
		{	logPrintf("Writing eigenvectors (VR) to '%s' ... ", evecFile.c_str()); logFlush();
			lbl.bcm->writeMatrix(VR, evecFile.c_str());
			logPrintf("done.\n");
		}
		#endif
	}
	else if(not ePhEnabled)
	{	//Simple probe-pump-probe with no relaxation:
		#ifdef PETSC_ENABLED
		lbl.report(-dt, lbl.drho);
		lbl.applyPump(); //takes care of optical pump or B-field excitation
		lbl.report(0., lbl.drho);
		#endif
	}
	else
	{	//Pump - time evolve - probe:
		#ifdef PETSC_ENABLED
		double tStart = 0.;
		bool checkpointExists = false;
		if(mpiWorld->isHead())
			checkpointExists = (checkpointFile.length()>0) and (fileSize(checkpointFile.c_str())>0);
		mpiWorld->bcast(checkpointExists);
		if(checkpointExists)
		{	logPrintf("Reading checkpoint from '%s' ... ", checkpointFile.c_str()); logFlush(); 
			//Determine offset of current process data and total expected file length:
			size_t offset = sizeof(int)+sizeof(double); //offset due to header
			for(int jProc=0; jProc<mpiWorld->iProcess(); jProc++)
				offset += sizeof(double) * lbl.rhoSize[jProc]; //offset due to data from previous processes
			size_t fsizeExpected = offset;
			for(int jProc=mpiWorld->iProcess(); jProc<mpiWorld->nProcesses(); jProc++)
				fsizeExpected += sizeof(double) * lbl.rhoSize[jProc];
			mpiWorld->bcast(fsizeExpected);
			//Open check point file and rrad time header:
			MPIUtil::File fp; mpiWorld->fopenRead(fp, checkpointFile.c_str(), fsizeExpected);
			mpiWorld->fread(&(lbl.stepID), sizeof(int), 1, fp);
			mpiWorld->fread(&tStart, sizeof(double), 1, fp);
			mpiWorld->bcast(tStart);
			//Read density matrix from check point file:
			mpiWorld->fseek(fp, offset, SEEK_SET);
			mpiWorld->fread(lbl.drho.data(), sizeof(double), lbl.drho.size(), fp);
			mpiWorld->fclose(fp);
			logPrintf("done.\n");
		}
		else
		{	//Do an initial report akin to above and apply the pump/B-field:
			lbl.report(-dt, lbl.drho);
			lbl.applyPump(); //takes care of optical pump or B-field excitation
			tStart = 0.; //integrate will report at t=0 below, before evolving ePh relaxation
		}
		//Evolve:
		if(tStep) //Fixed-step integrator:
			lbl.integrateFixed(lbl.drho, tStart, tStop, tStep, dt);
		else //Adaptive integrator:
			lbl.integrateAdaptive(lbl.drho, tStart, tStop, tolAdaptive, dt);
		#endif
	}
	
	//Cleanup:
	#ifdef PETSC_ENABLED
	CHECKERR(lbl.cleanup());
	if(not spectrumMode) CHECKERR(PetscFinalize());
	#endif
	FeynWann::finalize();
	return 0;
}
