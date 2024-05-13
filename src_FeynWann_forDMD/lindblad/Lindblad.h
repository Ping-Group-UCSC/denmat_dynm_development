/*-------------------------------------------------------------------
Copyright 2021 Ravishankar Sundararaman

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

#ifndef FEYNWANN_LINDBLAD_H
#define FEYNWANN_LINDBLAD_H

#include <core/matrix.h>
#include <lindblad/LindbladFile.h>
#include <Integrator.h>
#include <BlockCyclicMatrix.h>

#ifdef PETSC_ENABLED
	#include <petsc.h>
	
	//Replacement for CHKERRQ() macro in Petsc:
	#define CHECKERR(codeLine) \
		{	PetscInt iErr = codeLine; \
			if(iErr) \
			{	char* errMessage; \
				PetscErrorMessage(iErr, NULL, &errMessage); \
				die("PETSc error: %s\n\n", errMessage); \
			} \
		}
#else
	#define PetscErrorCode int
	#define CHECKERR(codeLine) codeLine;
#endif


//! Handling of valley degrees of freedom
enum ValleyMode
{	ValleyNone, //!< no special handling of valley
	ValleyInter, //!< only include inter-valley processes
	ValleyIntra //!< only include intra-valley processes
};


//Input parameters controlling Lindblad dynamics (implemented in Lindblad.cpp)
struct LindbladParams
{
	double dmu; //!< Fermi level position relative to neutral value / VBM
	double T; //!< Temperature
	
	double dt, tStop; //!< Time evolution reporting interval and end time
	double tStep; //!< Explicit time evolution step size (fixed integrator)
	double tolAdaptive; //!< Adaptive integrator relative tolerance

	double pumpOmega, pumpI; //!< pump frequency and intensity(CW)/flux(pulse)
	double pumpA0_sq() const; //!< compute the vector potential corresponding to pumpI and pumpOmega
	double pumpTau, pumpSigma; //!< pulse width (zero for CW) and energy conservation width
	vector3<complex> pumpPol; //!< pump polarization
	bool pumpEvolve; //!< whether pump is explicitly evolved in time
	
	bool pumpBfield; //!< whether the "pump" is a magnetic field initialization
	vector3<> pumpB; //!< initialization magnetic field
	vector3<> Bext; //!< constant external magnetic field applied post-initialization
	bool orbitalZeeman; //!< whether to include orbital zeeman coupling with magnetic fields

	vector3<> spinEchoB; //!< rotating magnetic field at t=0 in spin echo measurement (rotates about Bext)
	double spinEchoDelay; //!< time delay between pi/2 and pi pulses in spin echo setup
	double spinEchoOmega; //!< Larmor precession frequency for spin echo (if zero, set based on Bext)

	double omegaMin, domega, omegaMax; //!< probe frequency grid
	double tau; //!< probe width
	std::vector<vector3<complex>> pol; //!< probe polarizations
	double dE; //!< energy resolution for distribution functions

	bool linearized; //!< Whether dynamics is linearized
	bool spectrumMode; //!< Time-evolution spectrum if yes, dynamics otherwise
	int blockSize; //!< block size in ScaLAPACK matrix for spectrum mode only
	#ifdef SCALAPACK_ENABLED
	BlockCyclicMatrix::DiagMethod diagMethod;
	#endif

	bool ePhEnabled; //!< whether e-ph coupling is enabled
	double defectFraction; //!< defect fraction if present
	ValleyMode valleyMode; //!< whether all k-pairs (None) or only those corresponding to Intra/Inter-valley scattering are included
	bool verbose; //!< whether to print more detailed stats during evolution
	bool saveDist; //!< whether to save distributions at each reporting step
	string inFile; //!< file name to get lindblad data from
	string checkpointFile; //!< file name to save checkpoint data to
	string evecFile; //!< filename to write eigenvectors to in spectrum mode

	//---- Dependent variables computed from above ----
	double invT; //!< inverse temperature
	double nomega; //!< number of probe frequencies

	double spinEchoFlipTime; //!< Flip time i.e. pi-pulse duration in spin echo setup
	matrix3<> spinEchoRot; //!< Rotation matrix that takes x, z axes to spinEchoB, Bext directions
	vector3<> spinEchoTransform(vector3<> v, double t) const; //!< convert v from lab to rotating frame at time t (or inverse at -t)
	vector3<> spinEchoGetB(double t) const; //!< Get time-dependent magnetic field for spin echo measurement

	void initialize(); //!< Set dependent variables
};


// Base class of all lindblad dynamics (implemented in Lindblad.cpp and LindbladDynamics.cpp)
class Lindblad : public Integrator<DM1>
{
protected:
	const LindbladParams& lp;
	int stepID; //current time and reporting step number
	
	bool spinorial; //!< whether spin is available
	int spinWeight; //!< weight of spin in BZ integration
	matrix3<> R; double Omega; //!< lattice vectors and unit cell volume

	size_t nk, nkTot; //!< number of selected k-points overall and original total k-points effectively used in BZ sampling
	size_t ikStart, ikStop, nkMine; //!< range and number of selected k-points on this process
	TaskDivision kDivision;
	inline bool isMine(size_t ik) const { return kDivision.isMine(ik); } //!< check if k-point index is local
	inline int whose(size_t ik) const { return kDivision.whose(ik); } //!< find out which process (in mpiWorld) this k-point belongs to

	struct State : LindbladFile::Kpoint
	{	int ik; //global index of current k
		int innerStop; //end of active inner window range (relative to outer window)
		diagMatrix rho0; //equilibrium / initial density matrix (diagonal)
		matrix pumpPD; //P matrix elements at pump polarization x energy conservation delta (D), but without A0 and time factor
		
		//Handling of off-diagonal constant perturbations built into effective H0:
		matrix V0; //if non-null, eigenvectors of H0 (i.e. H0 is not diagonal)
		diagMatrix E0; //eigenvalues of H0 (equal to, or perturbed relative to E[innerStart:innerStop])

		matrix phase; //exp(-i H0 t) phases / unitary transform from interaction to Schrodinger picture
		matrix drho; //change of matrix from rho0 (Schrodinger picture)
		matrix rho; //density matrix in current state of the system (Schrodinger picture)
		matrix rhoDot; //contribution to drho/dt at current state (Schrodinger picture, before a global +H.C. added in getStateDot())
	};
	std::vector<State> state; //!< all information read from lindbladInit output (e and e-ph properties) + extra local variables above
	std::vector<int> nInnerAll; //!< nInner for all k-points on all processes
	std::vector<int> isKall; //whether each k-point is closer to K or K' (used only if valleyMode is not ValleyNone)
	double Emin, Emax; //!< energy range of active space across all k (for spin and number density output)
	
	//---- Flat density matrix storage and access functions ----
	DM1 drho; //!< flat array of change in density matrices (relative to rho0) of all k stored on this process
	std::vector<double> Eall; //!< inner window energies for all k (only needed and initialized when ePhEnabled)
	std::vector<size_t> nInnerPrev; //!< cumulative nInner for each k, which is the offset into the Eall array for each k
	std::vector<size_t> nRhoPrev; //cumulative nInner^2 for each k, which is the offset into the global rho structure for each k
	std::vector<size_t> rhoOffset; //!< array of offsets into process's rho for each k
	std::vector<size_t> rhoSize; //!< total size of rho on each process
	size_t rhoOffsetGlobal; //!< offset of current process rho data in the overall data
	size_t rhoSizeTot; //!< total size of rho
	inline matrix getRho(const double* rhoData, int N) const; //!< Get an NxN complex Hermitian matrix from a real array of length N^2
	inline void accumRho(const diagMatrix& in, double* rhoData) const; //!< Accumulate a diagonal matrix to a real array of length N^2
	inline void accumRhoHC(const matrix& in, double* rhoData) const; //!< Accumulate NxN matrix + its H.C. to a real array of length N^2
	
	const vector3<> K, Kp; //!< K and K' valley in reciprocal lattice coordinates
	static inline vector3<> wrap(const vector3<>& x); //!< Wrap fratcional coordinates to fundamental interval
	inline bool isKvalley(vector3<> k) const { return (wrap(K-k)).length_squared() < (wrap(Kp-k)).length_squared(); }

	//Scrodinger picture (SP) <-> interaction picture (IP) interface (in LindbladDynamics.cpp):
	void setState(double t, const DM1& drho, State& s) const; //!< IP drho -> SP s.drho (also zero out s.rhoDot)
	void getStateDot(const State& s, DM1& rhoDot) const; //!< SP s.rhoDot -> IP rhoDot (at t from last setState())

	//Interface to contributions implemented in subclasses (works on data within state):
	virtual void rhoDotScatter() = 0; //overall scattering contribution coupling all states

public:
	Lindblad(const LindbladParams& lp);
	virtual ~Lindblad() {}

	//I/O routines:
	bool readCheckpoint(double& t); //!< read checkpoint file; set final t and return true if state loaded
	void writeCheckpoint(double t) const; //!< write checkpoint file corresponding to time t
	void writeImEps(string fname) const; //!< Write probe response at current rho

	//Overall calculation and integrator interaction (in LindbladDynamics.cpp):
	virtual void calculate(); //!< set up initial state and run dynamics as specified
	void applyPump(); //!< one-shot pump (optical or Bfield)
	DM1 compute(double t, const DM1& v); //specify differential equation for time evolution
	void report(double t, const DM1& v) const; //called by integrator for periodic reporting
	void reportCarrierLifetime() const; //calculate and report fprime-averaged carrier lifetime
};


//Full nonlinear implementation of real-time Lindblad dynamics (in LindbladNonlinear.cpp)
class LindbladNonlinear : public Lindblad
{
public:
	LindbladNonlinear(const LindbladParams& lp);
	virtual ~LindbladNonlinear() {}

protected:
	virtual void rhoDotScatter();
};


//Base class of lindblad implementations with explicit time-evolution matrix (in LindbladMatrix.cpp)
class LindbladMatrix : public Lindblad
{
protected:
	void initializeMatrix();
	LindbladMatrix(const LindbladParams& lp) : Lindblad(lp) {}
	virtual ~LindbladMatrix() {}
};


//Linearized real-time Lindblad dynamics (in LindbladLinear.cpp)
class LindbladLinear : public LindbladMatrix
{	friend class LindbladMatrix;
	std::vector<int> nnzD, nnzO; //!< number of process-diagonal and process off-diagonal entries by row
	#ifdef PETSC_ENABLED
	Mat evolveMat; //!< Time evolution operator
	Vec vRho, vRhoDot; //!< temporary copies of drho and rdhoDot data in Petsc format
	void initialize(); //Initialize Petsc library
	void cleanup(); //Clean up Petsc quantities and library
	#endif

public:
	LindbladLinear(const LindbladParams& lp);
	virtual ~LindbladLinear();

protected:
	virtual void rhoDotScatter();
};


//Diagonalization of Lindblad superoperator to get time evolution spectrum (in LindbladSpectrum.cpp)
class LindbladSpectrum : public LindbladMatrix
{	friend class LindbladMatrix;
	std::vector<std::vector<std::pair<double,int>>> evolveEntries; //intermediate matrix elements by target process
	#ifdef SCALAPACK_ENABLED
	std::shared_ptr<BlockCyclicMatrix> bcm;  //Block-cyclic matrix descriptor
	BlockCyclicMatrix::Buffer evolveMat, spinMat, spinPert; //Time evolution, spin and spin perturbation matrices
	#endif

public:
	LindbladSpectrum(const LindbladParams& lp);
	virtual ~LindbladSpectrum() {}
	virtual void calculate(); //override dynamics with spectrum calculation

protected:
	virtual void rhoDotScatter() {} //Not used; no dynamics
};


//----- Inline function implementations -----

inline vector3<> Lindblad::wrap(const vector3<>& x)
{	vector3<> result = x;
	for(int dir=0; dir<3; dir++)
		result[dir] -= floor(0.5 + result[dir]);
	return result;
}


//Get an NxN complex Hermitian matrix from a real array of length N^2
inline matrix Lindblad::getRho(const double* rhoData, int N) const
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
inline void Lindblad::accumRho(const diagMatrix& in, double* rhoData) const
{	const int N = in.nRows();
	for(int i=0; i<N; i++)
	{	*(rhoData) += in[i];
		rhoData += (N+1); //advance to next diagonal entry
	}
}


//Accumulate an NxN matrix and its Hermitian conjugate to a real array of length N^2
inline void Lindblad::accumRhoHC(const matrix& in, double* rhoData) const
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


//Dot product for contracting dipole matrix elements to specified polarization:
inline matrix dot(const matrix* P, vector3<complex> pol)
{	return pol[0]*P[0] + pol[1]*P[1] + pol[2]*P[2];
}


//Construct identity - X:
inline diagMatrix bar(const diagMatrix& X)
{	diagMatrix Xbar(X);
	for(double& x: Xbar) x = 1. - x;
	return Xbar;
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

#endif
