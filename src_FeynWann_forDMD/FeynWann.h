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

#ifndef FEYNWANN_FEYNWANN_H
#define FEYNWANN_FEYNWANN_H

#include "DistributedMatrix.h"

//! Parameters for initializing Wannier
struct FeynWannParams
{	int iSpin; //!< current spin channel: 0(up) or 1(dn) for z-spin calculations; must be 0 for all other spin types
	string totalEprefix; //!< filename prefix for DFT outputs (default: Wannier/totalE)
	string phononPrefix; //!< filename prefix for phonon outputs (default: Wannier/phonon)
	string wannierPrefix; //!< filename prefix for wannier outputs (default: Wannier/wannier)
	bool needSymmetries; //!< whether to read symmetries from .sym file from JDFTx (default: false)
	bool needPhonons; //!< whether to initialize phonon-related quantities (default: false)
	bool needVelocity; //!< whether to initialize velocity (momentum) matrix elements
	bool needSpin; //!< whether to initialize spin matrix elements (will be reset to false if not relativistic)
	bool needL; //!< whether to initialize angular momentum matrix elements
	bool needQ; //!< whether to initialize r*p electric quadrupole matrix elements
	bool needLinewidth_ee; //!< whether to provide e-e line-width (default: false)
	bool needLinewidth_ePh; //!< whether to provide e-ph line-width (default: false)
	bool needLinewidthP_ePh; //!< whether to provide momentum-relaxation e-ph line-width (default: false)
	bool ePhHeadOnly; //!< if true, only evaluate ePh callback function at head of each offset for debugging (default: false)
	bool maskOptimize; //!< if true, optimize for heavily masked loops by switching between transform and compute based on a benchmark performed at startup
	bool bandSumLQ; //!< if true, use sum over bands to compute L and Q
	
	static const std::vector<double> fGrid_ePh; //!< fillings grid used for e-ph linewidths
	
	string needDefect; //!< if non-null, read matrix elements for defect with name specified by this string
	string needLinewidth_D; //!< if non-null, provide e-defect line-width for defect with name specified by this string
	string needLinewidthP_D; //!< if non-null, provide e-defect momentum-relaxation line-width for defect with name specified by this string
	
	vector3<> Bext; //!< external magentic field (added as a Zeeman perturbation to hamiltonian in FeynWann:setState)
	bool orbitalZeeman; //!< whether to include orbital magnetic moment from L in Zeeman perturbation
	double EzExt; //!< external electric field (added as a Stark perturbation to hamiltonian in FeynWann:setState)
	double scissor; //!< scissor operator to move conduction band states up in energy to fix band gap in post-processing
	double EshiftWeight; //!< if non-zero, apply this energy shift to the region of space selected by wannier slab weight (in mlwfW)
	bool enforceKramerDeg; //!< whether to enforce Kramer degeneracy in eigenvalues	
	double degeneracyThreshold; //!< threshold within which to treat Wannier bands as degenerate
	bool tdep; //!< use TDEP interface to compute phonon frequencies and eigenvectors

	FeynWannParams(class InputMap* inputMap=0); //!< If specified, look for optional parameters Bext (in Tesla), EzExt (in eV/nm), scissor (in eV) and EshiftWeight (in eV) from in inputMap
	void printParams() const; //!< Print the parameters read from inputMap (in atomic units)
	
	inline bool needRP() const { return (needL or needQ) and (not bandSumLQ); } //!< whether Wannierized RP matrix elements are needed
	
	//! Apply scissor operator correction to eigenvalues
	inline void applyScissor(diagMatrix& E) const
	{	for(double& Ei: E)
			if(Ei > degeneracyThreshold)
				Ei += scissor;
	}

	//JX
	bool needLayer, trunc_iDi_outer;
	string Bin_model; double alpha_Bin, kcut_model, kcut_width_model; vector3<> dir_2drb, pshk, pshs;
	int bStart_model, bEnd_model; bool renorm_Bin;
	double scissor_above_band; int band_scissor;
	//JX
};

//! Wannier interpolator for electrons and phonons
class FeynWann
{
public:
	static InitParams initialize(int argc, char** argv, const char* description); //!< wrap initSystemCmdLine from JDFTx
	static void finalize(); //!< wrap finalizeSystem from JDFTx
	static vector3<> randomVector(MPIUtil* mpiUtil=0); //!< uniformly random vector in [0,1)^3, constant across mpi instance, if any

	FeynWannParams& fwp;//JX
	FeynWann(FeynWannParams& fwp);
	void free(); //!< free matrices
	inline bool isRelativistic() const { return nSpinor==2; }
	
	//! Electronic properties at a given wave vector
	struct StateE
	{	int ik; //!< index in mesh of dimensions offsetDim
		vector3<> k; //!< wave-vector in recip lattice coords
		diagMatrix E; //!< energy relative to Fermi level (FeynWann::mu)
		matrix U; //!< rotation from Wannier to eiegen-basis
		matrix v[3]; //!< velocity matrix elements in Cartesian coordinates, available if needVelocity = true
		std::vector<vector3<>> vVec; //!< band velocities (diagonal part of v) in Cartesian coordinates, available if needVelocity = true
		matrix S[3]; //!< Spin matrix elements in Cartesian coordinates, available if needSpin = true
		std::vector<vector3<>> Svec; //!< band spins (diagonal part of S) in Cartesian coordinates, available if needSpin = true
		matrix L[3]; //!< Angular momentum matrix elements in Cartesian coordinates, available if needL = true
		matrix Q[5]; //!< Electric quadrupole r*p matrix elements (xy, yz, zx, xxr, yyr), available if needQ = true
		diagMatrix ImSigma_ee; //!< e-e linewidth, available if needLinewidth_ee = true
		double ImSigma_ePh(int n, double f) const; //!< get e-ph linewidth for band n given its occupation f, available if needLinewidth_ePh = true
		double ImSigmaP_ePh(int n, double f) const; //!< get e-ph linewidth for band n given its occupation f, available if needLinewidthP_ePh = true
		diagMatrix ImSigma_D; //!< e-defect linewidth, available if needLinewidth_D is set
		diagMatrix ImSigmaP_D; //!< e-defect momentum-relaxing linewidth, available if needLinewidthP_D is set
		matrix dHePhSum, layer;//JX dHePhSum - nBands*nBands x 3 matrix used internally to enforce e-ph matrix element sum rule at all k's; layer - layer occupation
	private:
		std::vector<diagMatrix> logImSigma_ePhArr; //!< e-ph linewidth for each f in fGrid_ePh
		std::vector<diagMatrix> logImSigmaP_ePhArr; //!< e-ph momentum-relaxation linewidth for each f in fGrid_ePh
		//matrix dHePhSum;//JX //!< nBands*nBands x 3 matrix used internally to enforce e-ph matrix element sum rule at all k's
		bool withinRange; //!< whether any bands in E are within ePhEstart and ePhEstop (used to filter ePhLoop)
		matrix getMatrixRotated(const std::shared_ptr<DistributedMatrix>& mat, int iMat=0) const;
		void computeLQ(const FeynWannParams& fwp,
			const std::shared_ptr<DistributedMatrix> RPw,
			const std::shared_ptr<DistributedMatrix> HprimeW[3]); //!< calculate L and Q
		void compute_dHePhSum(const std::shared_ptr<DistributedMatrix> Dw,
			const std::shared_ptr<DistributedMatrix> HePhSumW); //!< calculate phonon sum rule correction
		static void extractDiagonal(const matrix (&X)[3], std::vector<vector3<>>& Xvec); //!< used to initialize vVev, Svec
		friend class FeynWann;
	};
	
	//! Phonon properties at a given wave vector
	struct StatePh
	{	int iq; //!< index in mesh of dimensions offsetDim
		vector3<> q; //!< wave-vector in recip lattice coords
		diagMatrix omega; //!< frequency
		matrix U; //!< rotation from atom-displacement to eigen-basis
	};
	
	//! Electron-phonon matrix elements
	struct MatrixEph
	{	const StateE* e1; //!< corresponding first electronic state
		const StateE* e2; //!< corresponding second electronic state
		const StatePh* ph; //!< corresponding phonon state
		std::vector<matrix> M; //!< nModes matrices of nBands x nBands matrix elements
	};
	
	//! Electron-defect matrix elements
	struct MatrixDefect
	{	const StateE* e1; //!< corresponding first electronic state
		const StateE* e2; //!< corresponding second electronic state
		matrix M; //!< nBands x nBands matrix elements
	};
	
	typedef void (*eProcessFunc)(const StateE& state, void* params); //!< Callback function pointer for eLoop()
	typedef void (*phProcessFunc)(const StatePh& state, void* params); //!< Callback function pointer for phLoop()
	typedef void (*ePhProcessFunc)(const MatrixEph& mat, void* params); //!< Callback function pointer for ePhLoop()
	typedef void (*defectProcessFunc)(const MatrixDefect& mat, void* params); //!< Callback function pointer for defectLoop()
	
	//! Calculate electronic properties for each k-point in a mesh offset by k0
	//! Calls provided callback function eProcess on each of them, along with provided params
	//! Optional array mask selects which indices within the offset to actually calculate, skipping the rest for efficiency.
	void eLoop(const vector3<>& k0, eProcessFunc eProcess, void* params, const std::vector<bool>* mask=0);
	void eCalc(const vector3<>& k, StateE& e); //!< Calculate electronic properties for a single k and store results in e on group head
	void eTransformNeeded(const vector3<>& k0); //!< Helper to transform all needed matrix elements at offset k0
	void eComputeNeeded(const vector3<>& k); //!< Helper to compute all needed matrix elements at single k
	size_t eCountPerOffset() const { return Hw->nkTot; } //!< number of k's sampled per offset = prod(offsetDim)
	
	//! Calculate phonon properties for each q-point in a mesh offset by q0
	//! Calls provided callback function phProcess on each of them, along with provided params
	void phLoop(const vector3<>& q0, phProcessFunc phProcess, void* params);
	void phCalc(const vector3<>& q, StatePh& ph); //!< Calculate phonon properties for a single q and store results in ph on group head
	size_t phCountPerOffset() const { return OsqW->nkTot; } //!< number of q's sampled per offset = prod(offsetDim)
	
	//! Calculate electronic properties for each pair of k-points between two meshes offset by k01 and k02,
	//! as well as phonon properties and electron-phonon matrix elements connecting these k-points.
	//! Calls provided callback function ePhProcess on each of them, along with provided params
	//! Optionally invoke non-null eProcess and phProcess callback functions (with same params) before the ePhProcess call back function,
	//! which is effectively a more efficient way of calling eLoop's and phLoop with the same offsets beforehand.
	//! Optional masks eMask1 and eMask2 for the two electronic offsets, and ePhMask select the active k / k-pairs to compute and skip the rest for efficiency.
	//! Note that ePhMask is of dimensions eCountPerOffset^2, with outer loop over ik1 and inner loop over ik2.
	void ePhLoop(const vector3<>& k01, const vector3<>& k02, ePhProcessFunc ePhProcess, void* params,
		eProcessFunc eProcess1=0, eProcessFunc eProcess2=0, phProcessFunc phProcess=0,
		const std::vector<bool>* eMask1=0, const std::vector<bool>* eMask2=0, const std::vector<bool>* ePhMask=0);
	void ePhCalc(const StateE& e1, const StateE& e2, const StatePh& ph, MatrixEph& m); //!< Calculate e-ph matrix elements coupling e1, e2 and ph and store it in m on group head
	size_t ePhCountPerOffset() const { return size_t(Hw->nkTot) * size_t(Hw->nkTot); } //!< number of k-pairs sampled per offset = prod(offsetDim)^2
	
	//Similar functions for defects:
	void defectLoop(const vector3<>& k01, const vector3<>& k02, defectProcessFunc defectProcess, void* params,
		eProcessFunc eProcess1=0, eProcessFunc eProcess2=0,
		const std::vector<bool>* eMask1=0, const std::vector<bool>* eMask2=0, const std::vector<bool>* defectMask=0);
	void defectCalc(const StateE& e1, const StateE& e2, MatrixDefect& m); //!< Calculate e-defect matrix elements coupling e1 and e2, and store it in m on group head
	size_t defectCountPerOffset() const { return size_t(Hw->nkTot) * size_t(Hw->nkTot); } //!< number of k-pairs sampled per offset = prod(offsetDim)^2
	
	void symmetrize(matrix3<>& m) const; //!< symmetrize a tensor in Cartesian coordinates (available if needSymmetries = true)
	
	//DFT / Wannier / Phonon parameters:
	matrix3<> R; //!< lattice vectors
	std::vector<vector3<>> atpos; //!< atomic positions
	std::vector<string> atNames; //!< atom species names
	int nAtoms; //number of atoms
	double Omega; //!< unit cell volume
	vector3<int> kfold; //!< k-point folding in original calculation
	vector3<int> phononSup; //!< phonon supercell in original calculation
	vector3<int> kfoldSup; //!< k-point folding of phonon supercell i.e. kfold / phononSup
	vector3<int> offsetDim; //!< k/q mesh dimensions associated with offset = kfold without phonons and phononSup with phonons
	std::vector<vector3<>> qOffset; //!< list of offsets to q-mesh requred to cover k-mesh (of length = prod(kfoldSup))
	vector3<bool> isTruncated; //!< whether each direction is truncated
	std::vector<SpaceGroupOp> sym; //!< symmetries of DFT calculation
	int nBands; //!< number of Wannier bands for the electrons
	int nSpins, nSpinor, spinWeight; //!< number of spin channels, spinor components and weight per spin channel
	bool realPartOnly; //!< whether wannier Hamiltonians / matrix elements in files are stored with real parts alone
	string spinSuffix; //filename suffix for current spin channel (if any)
	double mu; //!< chemical potential if DFT calculation had smearing, else VBM
	double nElectrons; //!< number of electrons per unit cell in DFT calculation
	double EminInner, EmaxInner; //!< energy range for inner window (within which eigenvalues should be exact compared to DFT)
	int nModes; //!< number of phonon modes (polarizations)
	
	//Electrons:
	std::vector<vector3<int>> cellMap; //electron Wannier cell map
	std::vector<matrix> cellWeights; //corresponding weights (nBands x nBands for each cell)
	std::shared_ptr<DistributedMatrix> Hw, Pw, Sw, RPw, Zw; //Wannier hamiltonian, momentum, spin, R*P and z matrix elements
	std::shared_ptr<DistributedMatrix> HprimeW[3]; //d/dk of Wannier hamiltonian in each Cartesian direction
	std::shared_ptr<DistributedMatrix> ImSigma_eeW, ImSigma_ePhW, ImSigmaP_ePhW, ImSigma_DW, ImSigmaP_DW; //linewidths in wannier basis
	void setState(StateE& state); //!< set requested properties for ik in state
	void bcastState(StateE& state, MPIUtil* mpiUtil, int root); //!< broadcast specified state on specified MPI instance
	
	//Phonons:
	std::vector<vector3<int>> phononCellMap; //cell map for phonon force matrix
	diagMatrix invsqrtM; //!< 1/sqrt(M) per nuclear displacement mode
	bool polar; //!< whether the system is polar (i.e. needs LO-TO correction)
	std::vector<vector3<>> Zeff; //Born effective charge for polar materials
	matrix3<> epsInf; // epsilon at infinity for polar material
 	std::vector<vector3<>> epsInf2D; // epsilon at infinity for polar material (2D case has extra row to 3x3 epsilon inf)
 	int truncDir; // save truncation dir for calling lrs2D
 	double omegaEff; //2D area for lrs2D 
	std::shared_ptr<class LongRangeSum> lrs; //long range sum corrector for e-ph matrix elements
	std::shared_ptr<class LongRangeSum2D> lrs2D; //long range sum corrector for e-ph matrix elements (2D)
	matrix phononCellWeights; //corresponding weights (nAtoms*nAtoms x nCells), available if polar
	std::shared_ptr<DistributedMatrix> OsqW; //phonon omega-squared matrix
	void setState(StatePh& state); //!< set requested properties for iq in state
	void bcastState(StatePh& state, MPIUtil* mpiUtil, int root); //!< broadcast specified state on specified MPI instance
	
	//Electron-phonon interaction:
	std::vector<vector3<int>> ePhCellMap; //cell map for e-ph matrix elements
	std::vector<vector3<int>> ePhCellMapSum; //cell map for e-ph matrix element sum rule
	std::shared_ptr<DistributedMatrix> HePhW; //electron-phonon matrix elements in Wannier basis
	std::shared_ptr<DistributedMatrix> HePhSumW; //electron-phonon matrix element sum rule in Wannier basis
	std::shared_ptr<DistributedMatrix> Dw; //gradient matrix elements for sum rule enforcement
	double ePhEstart, ePhEstop; //energy range restriction of ePhloop, enabled if ePhEstart < ePhEstop
	int tTransformByCompute; //benchmark ratio of transform time to compute time used to switch between transform and compute in ePhLoop (if maskOptimize = true)
	void setMatrix(const StateE& e1, const StateE& e2, const StatePh& ph, int ikPair, MatrixEph& m); //set e-ph properties for e1.ik, e2.ik and ph.iq in m
	
	//Electron-defect interaction:
	vector3<int> defectSup; //supercell size of exported e-defect matrix elements
	std::vector<vector3<int>> defectCellMap; //cell map for e-defect matrix elements
	std::shared_ptr<DistributedMatrix> HdefectW; //electron-defect matrix elements in Wannier basis
	int tTransformByComputeD; //benchmark ratio of transform time to compute time used to switch between transform and compute in ePhLoop (if maskOptimize = true)
	void setMatrix(const StateE& e1, const StateE& e2, int ikPair, MatrixDefect& m); //set defect properties for e1.ik and e2.ik in m
	
	//JX
	matrix3<> G, GT, GGT;
	int nBandsDFT, nStatesDFT;
	bool isMetal;
	std::shared_ptr<DistributedMatrix> Layerw; //Wannier layer matrix elements
	void bcastState_inEphLoop(StateE& state, MPIUtil* mpiUtil, int root); //!< broadcast specified state on specified MPI instance
	void bcastState_JX(StateE& state, MPIUtil* mpiUtil, int root, bool need_dHePhSum); //!< broadcast specified state on specified MPI instance
	bool energyOnly;//if true, setState will only interpolation E and U
	vector3<> Bso;
	matrix3<> gfac;//g-factor tensor. Diagonal and about 2 by default
	void copy_stateE(const StateE& e, StateE& eout);
	void trunc_stateE(StateE& e, StateE& eTrunc, int b0_eph, int b1_eph, int b0_dm, int b1_dm, int b0_probe, int b1_probe);
	double omegaPhCut;
	bool is_state_withinRange(StateE& e);
	vector3<> calc_Bso(vector3<> kvec);
	matrix restrictBandRange(const matrix& mat, const int& bStart, const int& bEnd) const;
	std::vector<vector3<>> phononBasis;
	//JX

private:
	bool inEphLoop; //flag used internally by setState etc. for special handling of sum rule quantities within an ePhLoop
	std::shared_ptr<MPIUtil> mpiInterGroup; //inter-group communicator used for matrix element initialization
	std::shared_ptr<DistributedMatrix> readE(string varname, int nVars=1) const; //read electronic matrix elements (vetcor size nVars)
	matrix restrictInnerWindow(const matrix& mat, const diagMatrix& E) const; //prokect out contributions beyond the inner window
};

//Utility functions for printing with error estimates:
void reportResult(const std::vector<matrix3<>>& result, string resultName, double unit, string unitName, FILE* fp=globalLog, bool invAvg=false); //!< report a tensor result with error bars
void reportResult(const std::vector<vector3<>>& result, string resultName, double unit, string unitName, FILE* fp=globalLog, bool invAvg=false); //!< report a vector result with error bars
void reportResult(const std::vector<double>& result, string resultName, double unit, string unitName, FILE* fp=globalLog, bool invAvg=false); //!< report a scalar result with error bars

//Fermi and Bose functions with overflow/underflow handling:
inline double fermi(double EminusMuByT) //!< Fermi function with overflow/underflow handling
{	if(EminusMuByT < -46.) return 1.;//JX 36->46
	else if(EminusMuByT > 46.) return 0.;//JX 36->46
	else return 1./(1. + exp(EminusMuByT));
}
inline void fermi(double EminusMuByT, double& f, double& fbar) //!< version that computes both f and fbar=1-f without loss of precision
{	if(EminusMuByT < 0.)
	{	double boltz = exp(EminusMuByT);
		fbar = boltz / (1. + boltz);
		f = 1.-fbar;
	}
	else
	{	double boltz = exp(-EminusMuByT);
		f = boltz / (1. + boltz);
		fbar = 1.-f;
	}
}
inline double fermiPrime(double EminusMuByT) //!< Fermi function derivative w.r.t E/T (multiply by 1/T for df/dE)
{	if(fabs(EminusMuByT) > 46.) return 0.;//JX 36->46
	else return 0.25*(std::pow(tanh(0.5*EminusMuByT), 2) - 1.);
}
inline double bose(double omegaByT) //!< Bose function with overflow/underflow handling
{	if(omegaByT > 46.) return 0.;//JX 36->46
	else return 1./(exp(omegaByT) - 1.);
}

#endif //FEYNWANN_FEYNWANN_H
