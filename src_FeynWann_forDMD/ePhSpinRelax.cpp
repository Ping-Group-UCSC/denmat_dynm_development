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
#include "SparseMatrix.h"
#include <core/Units.h>
#include <core/Random.h>
#include <core/LatticeUtils.h>
#include <algorithm>

struct SpinRelaxCollect
{	const std::vector<double>& dmu; //doping levels
	const std::vector<double>& T; //temperatures
	const double omegaPhByTmin; //lower cutoff in phonon frequency (relative to T)
	const int nModes; //number of phonon modes to include in calculation (override if any, applied below already)
	
	const double degeneracyThreshold; //threshold for degenerate subspace projection
	const double sqrtEconserveExpFac; //exponential factor in sqrt of Gaussian delta for energy conservation
	const double prefacGamma, prefacChi; //prefactors for numerator and denominator of T1
	const double Estart, Estop; //energy range close enough to band edges or mu's to be relevant
	std::vector<matrix3<>> Gamma, chi; //numerator and denominator in T1^-1, for each T, dmu and formula (FGR vs dRho)
	
	//Quantities for evaluating inter-valley contributions
	const bool valley; //whether to evaluate inter-valley contributions
	const matrix3<> G, GGT;
	const vector3<> K, Kp;
	std::vector<matrix3<>> GammaV; //valley-weighted numerator of T1^-1
	
	SpinRelaxCollect(const std::vector<double>& dmu, const std::vector<double>& T, double omegaPhByTmin, int nModes, double degeneracyThreshold,
		double EconserveWidth, size_t nKpairs, size_t nKtot, double Estart, double Estop, bool valley, matrix3<> R)
	: dmu(dmu), T(T), omegaPhByTmin(std::max(1e-3,omegaPhByTmin)), nModes(nModes), degeneracyThreshold(degeneracyThreshold),
		sqrtEconserveExpFac(-0.25/std::pow(EconserveWidth, 2)),
		prefacGamma(2*M_PI/ (nKpairs * sqrt(2.*M_PI)*EconserveWidth)), //include prefactor of Gaussian energy conservation
		prefacChi(0.5/nKtot), //collected over both k1 and k2 arrays
		Estart(Estart), Estop(Estop),
		Gamma(T.size()*dmu.size()*2), chi(T.size()*dmu.size()*2),
		valley(valley), G(2*M_PI * inv(R)), GGT(G * (~G)), K(1./3, 1./3, 0), Kp(-1./3, -1./3, 0), GammaV(Gamma)
	{
	}
	
	inline SparseMatrix degenerateProject(const matrix& M, const diagMatrix& E, int bStart, int bStop)
	{	int bCount = bStop-bStart;
		SparseMatrix result(bCount, bCount, bCount); //nNZ estimate based on diagonal (Rashba)
		for(int b2=bStart; b2<bStop; b2++)
		{	const complex* Mdata = M.data() + (b2*M.nRows() + bStart);
			for(int b1=bStart; b1<bStop; b1++)
			{	if(fabs(E[b1] - E[b2]) < degeneracyThreshold)
				{	SparseEntry sr;
					sr.i = b1 - bStart; //relative to submatrix
					sr.j = b2 - bStart;
					sr.val = *(Mdata);
					result.push_back(sr);
				}
				Mdata++;
			}
		}
		return result;
	}
	
	//Helper functions for evaluating inter-valley weights (Hex crystals only):
	static inline vector3<> wrap(const vector3<>& x)
	{	vector3<> result = x;
		for(int dir=0; dir<3; dir++)
			result[dir] -= floor(0.5 + result[dir]);
        return result;
	}
	inline bool isKvalley(vector3<> k) const
	{	return GGT.metric_length_squared(wrap(K-k))
			< GGT.metric_length_squared(wrap(Kp-k));
	}
	
	//Narrow band range based on active energy range:
	void bandRangeNarrow(const FeynWann::StateE& e, int& bStart, int & bStop)
	{	for(int b=0; b<e.E.nRows(); b++)
		{	const double& E = e.E[b];
			if(E>=Estart and b<bStart) bStart=b;
			if(E<=Estop and b>=bStop) bStop=b+1;
		}
	}
	
	//Change in density matrix due to first order perturbation theory by Hamiltonian H1:
	inline matrix dRhoCalc(const diagMatrix& E, const diagMatrix& F, double T, const matrix& H1, int bStart, int bStop)
	{	int nBandsSel = bStop - bStart;
		matrix result(nBandsSel, nBandsSel); complex *rData = result.data();
		double invT = 1./T;
		for(int b2 = bStart; b2 < bStop; b2++)
			for(int b1 = bStart; b1 < bStop; b1++)
			{	double minus_fPrimeEff = 0.;
				if(fabs(E[b1] - E[b2]) <= degeneracyThreshold)
				{	double Favg = 0.5 * (F[b1] + F[b2]);
					minus_fPrimeEff =  Favg * (1. - Favg) * invT; //-df/dE
				}
				else minus_fPrimeEff = (F[b1] - F[b2]) / (E[b2] - E[b1]); //finite difference version of -df/dE
				*(rData++) = minus_fPrimeEff * H1(b1, b2);
			}
		return result;
	}
	
	//Calculate spin relaxation rate numerator (Gamma):
	void process(const FeynWann::MatrixEph& mEph)
	{	const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		const int nBands = e1.E.nRows();
		
		//Select relevant band range:
		int bStart1 = nBands, bStop1 = 0;
		int bStart2 = nBands, bStop2 = 0;
		bandRangeNarrow(e1, bStart1, bStop1);
		bandRangeNarrow(e2, bStart2, bStop2);
		int nBandsSel1 = bStop1 - bStart1; //reduced number of selected bands at this k-pair
		int nBandsSel2 = bStop2 - bStart2; //reduced number of selected bands at this k-pair
		int nBandsSelProd = nBandsSel1 * nBandsSel2;
		if(nBandsSelProd <= 0) return;
		
		//Degenerate spin projections:
		std::vector<SparseMatrix> Sdeg1(3), Sdeg2(3);
		std::vector<matrix> S1(3), S2(3);
		for(int iDir=0; iDir<3; iDir++)
		{	Sdeg1[iDir] = degenerateProject(e1.S[iDir], e1.E, bStart1, bStop1);
			Sdeg2[iDir] = degenerateProject(e2.S[iDir], e2.E, bStart2, bStop2);
			S1[iDir] = e1.S[iDir](bStart1,bStop1, bStart1,bStop1);
			S2[iDir] = e2.S[iDir](bStart2,bStop2, bStart2,bStop2);
		}
		
		//Weight for valley contrib
		double wValley = (valley and (isKvalley(e1.k) xor isKvalley(e2.k))) ? 1. : 0.;
		
		//Compute Gamma contributions by band pair and T, except for electron occupation factors:
		std::vector<std::vector<matrix3<>>> contribGamma(T.size(), std::vector<matrix3<>>(nBandsSelProd));
		std::vector<std::vector<matrix3<>>> contribGammaV(contribGamma); //for valley contributions
		std::vector<matrix> A(nModes); //phonon matrix elements * sqrt(energy conservation) for each phonon mode
		std::vector<vector3<matrix>> SAcomm(nModes); //commutator of A with spin matrices
		std::vector<diagMatrix> nPh(T.size(), diagMatrix(nModes)); //phonon occupation factors at each temperature
		std::vector<bool> modeContrib(nModes, false); //whether any Econserve factor non-zero for given phonon mode
		for(int alpha=0; alpha<nModes; alpha++)
		{	//Phonon occupation (nPh/T and prefactors) for each T:
			const double& omegaPh = ph.omega[alpha];
			std::vector<double> prefac_nPhByT(T.size());
			for(size_t iT=0; iT<T.size(); iT++)
			{	double invT = 1./T[iT];
				const double omegaPhByT = invT*omegaPh;
				if(omegaPhByT < omegaPhByTmin) continue; //avoid 0./0. below
				nPh[iT][alpha] = bose(omegaPhByT);
				prefac_nPhByT[iT] =  prefacGamma * invT * nPh[iT][alpha];
			}
			//Energy conservation factor by band pairs:
			const matrix G = mEph.M[alpha](bStart1,bStop1, bStart2,bStop2); //work only with sub-matrix of relevant bands
			A[alpha] = G; //G multiplied by sqrt(Econserve) below
			std::vector<double> Econserve(nBandsSelProd);
			int bIndex = 0;
			bool contrib = false;
			complex* Adata = A[alpha].data();
			for(int b2=bStart2; b2<bStop2; b2++)
			for(int b1=bStart1; b1<bStop1; b1++)
			{	double expTerm = sqrtEconserveExpFac * std::pow(e1.E[b1] - e2.E[b2] - omegaPh, 2);
				if(expTerm > -8)
				{	double sqrtEconserve = exp(expTerm); //compute exponential only when needed
					Adata[bIndex] *= sqrtEconserve;
					Econserve[bIndex] = sqrtEconserve * sqrtEconserve;
					modeContrib[alpha] = true;
					contrib = true;
				}
				else Adata[bIndex] = 0.;
				bIndex++;
			}
			if(not contrib) continue; //no energy conserving combination for this phonon mode at present k-pair
			//Calculate spin-phonon commutators:
			matrix SGcomm[3]; vector3<const complex*> SGcommData;
			for(int iDir=0; iDir<3; iDir++)
			{	SGcomm[iDir] = Sdeg1[iDir] * G - G * Sdeg2[iDir]; //for FGR version
				SGcommData[iDir] = SGcomm[iDir].data();
				SAcomm[alpha][iDir] = S1[iDir] * A[alpha] - A[alpha] * S2[iDir]; //for dRho version
			}
			//Collect commutator contributions for each T (FGR version):
			for(int bIndex=0; bIndex<nBandsSelProd; bIndex++) //loop over b2 and b1
			{	vector3<complex> SGcommCur = loadVector(SGcommData, bIndex);
				matrix3<> SGcommOuter = realOuter(SGcommCur, SGcommCur);
				for(size_t iT=0; iT<T.size(); iT++)
				{	matrix3<> contrib = (prefac_nPhByT[iT] * Econserve[bIndex]) * SGcommOuter;
					contribGamma[iT][bIndex] += contrib;
					if(valley) //inter-valley weighted contributions
						contribGammaV[iT][bIndex] += contrib * wValley;
				}
			}
		}
		
		//Collect results for various dmu values:
		for(size_t iT=0; iT<T.size(); iT++)
		{	double invT = 1./T[iT];
			for(size_t iMu=0; iMu<dmu.size(); iMu++)
			{	size_t iMuT = (iT*dmu.size() + iMu)*2; //combined index
				//Compute Fermi occupations and dRho perturbations:
				#define CALC_F_dRho(s) \
					diagMatrix F##s(nBands), Fbar##s(nBands); \
					vector3<matrix> dRho##s; \
					for(int b=bStart##s; b<bStop##s; b++) \
					{	fermi(invT*(e##s.E[b] - dmu[iMu]), F##s[b], Fbar##s[b]); \
					} \
					for(int jDir=0; jDir<3; jDir++) \
						dRho##s[jDir] = dRhoCalc(e##s.E, F##s, T[iT], e##s.S[jDir], bStart##s, bStop##s);
				CALC_F_dRho(1)
				CALC_F_dRho(2)
				#undef CALC_F_dRho
				
				//Accumulate Gamma contributions (FGR version):
				int bIndex = 0;
				for(int b2=bStart2; b2<bStop2; b2++)
				for(int b1=bStart1; b1<bStop1; b1++)
				{	Gamma[iMuT] += contribGamma[iT][bIndex] * (F2[b2] * Fbar1[b1]);
					if(valley) //inter-valley weighted result
						GammaV[iMuT] += contribGammaV[iT][bIndex] * (F2[b2] * Fbar1[b1]);
					bIndex++;
				}
				
				//Accumulate Gamma contributions (dRho version):
				for(int alpha=0; alpha<nModes; alpha++)
					if(modeContrib[alpha])
					{	matrix3<> contribGamma_dRho;
						diagMatrix nPhF1(bStop1-bStart1, nPh[iT][alpha]); nPhF1 += F1(bStart1,bStop1); //nPh + f1
						diagMatrix nPhFbar2(bStop2-bStart2, nPh[iT][alpha]); nPhFbar2 += Fbar2(bStart2,bStop2); //nPh + 1 - f2
						for(int jDir=0; jDir<3; jDir++)
						{	matrix dRhoAcomm_j = dRho1[jDir] * A[alpha] * nPhFbar2 - nPhF1 * A[alpha] * dRho2[jDir];
							for(int iDir=0; iDir<3; iDir++)
								contribGamma_dRho(iDir,jDir) = prefacGamma * dotc(SAcomm[alpha][iDir], dRhoAcomm_j).real();
						}
						Gamma[iMuT+1] += contribGamma_dRho;
						if(valley)
							GammaV[iMuT+1] += contribGamma_dRho * wValley;
					}
			}
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params)
	{	((SpinRelaxCollect*)params)->process(mEph);
	}

	//Calculate denominator chi (electron loop)
	void process(const FeynWann::StateE& e)
	{	const int nBands = e.E.nRows();
		
		//Select relevant band range:
		int bStart = nBands, bStop = 0;
		bandRangeNarrow(e, bStart, bStop);
		int nBandsSel = bStop - bStart; //reduced number of selected bands at this k-pair
		if(nBandsSel <= 0) return;
		
		//Degenerate spin projections:
		std::vector<SparseMatrix> Sdeg(3);
		for(int iDir=0; iDir<3; iDir++)
			Sdeg[iDir] = degenerateProject(e.S[iDir], e.E, bStart, bStop);
		
		//Compute chi contributions by band except for electron occupation factors:
		std::vector<matrix3<>> contribChi(nBands);
		for(int iDir=0; iDir<3; iDir++)
		for(int jDir=0; jDir<3; jDir++)
		{	diagMatrix SiSj = diagSS(Sdeg[iDir], Sdeg[jDir]);
			for(int b=bStart; b<bStop; b++)
				contribChi[b](iDir,jDir) = prefacChi * SiSj[b-bStart];
		}
	
		//Collect results for various dmu values:
		for(size_t iT=0; iT<T.size(); iT++)
		{	double invT = 1./T[iT];
			for(size_t iMu=0; iMu<dmu.size(); iMu++)
			{	size_t iMuT = (iT*dmu.size() + iMu)*2; //combined index
				//Compute Fermi occupations and accumulate chi contributions (FGR formula):
				diagMatrix F(nBands), Fbar(nBands);
				for(int b=bStart; b<bStop; b++)
				{	fermi(invT*(e.E[b] - dmu[iMu]), F[b], Fbar[b]);
					chi[iMuT] += (invT * F[b]*Fbar[b]) * contribChi[b];
				}
				//Accumulate chi contributions (dRho formula):
				matrix3<> contribChi_dRho;
				for (int jDir = 0; jDir < 3; jDir++)
				{	matrix dRho_j = dRhoCalc(e.E, F, T[iT], e.S[jDir], bStart, bStop);
					const complex* dRho_jData = dRho_j.data();
					for(int b2 = bStart; b2 < bStop; b2++)
					for(int b1 = bStart; b1 < bStop; b1++)
					{	for(int iDir = 0; iDir<3; iDir++)
							contribChi_dRho(iDir, jDir) += (e.S[iDir](b1, b2).conj() * (*dRho_jData)).real();
						dRho_jData++;
					}
				}
				chi[iMuT+1] += prefacChi * contribChi_dRho;
			}
		}
	}
	static void eProcess(const FeynWann::StateE& e, void* params)
	{	((SpinRelaxCollect*)params)->process(e);
	}

	//! Real part of outer product of complex vectors, Re(a \otimes b*):
	inline matrix3<> realOuter(const vector3<complex> &a, const vector3<complex> &b)
	{	matrix3<> m;
		for(int i=0; i<3; i++)
			for(int j=0; j<3; j++)
				m(i,j) = (a[i] * b[j].conj()).real();
		return m;
	}
};


//Helper class for collecting relevant energy range:
struct EnergyRangeCollect
{	const double &dmuMin, &dmuMax; //minimum and maximum chemical potentials considered
	double EvMax, EcMin; //VBM and CBM estimates
	double omegaPhMax; //max phonon energy
	
	EnergyRangeCollect(const std::vector<double>& dmu)
	: dmuMin(dmu.front()), dmuMax(dmu.back()),
		EvMax(-DBL_MAX), EcMin(+DBL_MAX), omegaPhMax(0.)
	{
	}
	
	void process(const FeynWann::StateE& state)
	{	const double tol = 1e-3;
		for(const double& E: state.E)
		{	if(E<dmuMin+tol and E>EvMax) EvMax = E;
			if(E>dmuMax-tol and E<EcMin) EcMin = E;
		}
	}
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	((EnergyRangeCollect*)params)->process(state);
	}
	
	static void phProcess(const FeynWann::StatePh& state, void* params)
	{	double& omegaPhMax = ((EnergyRangeCollect*)params)->omegaPhMax;
		omegaPhMax = std::max(omegaPhMax, state.omega.back());
	}
};


int main(int argc, char** argv)
{	InitParams ip = FeynWann::initialize(argc, argv, "Electron-phonon scattering contribution to spin relaxation.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const int nBlocks = inputMap.get("nBlocks"); assert(nBlocks>0);
	const double neglectThreshold = inputMap.get("neglectThreshold", 1e-8); //relative threshold in occupation and energy conservation factors that can be neglected (determines Emargin)
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const double Tmin = inputMap.get("Tmin") * Kelvin; //temperature; start of range
	const double Tmax = inputMap.get("Tmax", Tmin/Kelvin) * Kelvin; assert(Tmax>=Tmin); //temperature; end of range (defaults to Tmin)
	const size_t Tcount = inputMap.get("Tcount", 1); assert(Tcount>0); //number of temperatures
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", dmuMin/eV) * eV; assert(dmuMax>=dmuMin); //optional shift in chemical potential from neutral value; end of range (defaults to dmuMin)
	const size_t dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts (default 1)
	const double omegaPhByTmin = inputMap.get("omegaPhByTmin", 1e-3); //lower cutoff in phonon frequency (relative to temperature)
	const int nModesOverride = inputMap.get("nModesOverride", 0); //if non-zero, use only these many lowest phonon modes (eg. set to 3 for acoustic only in 3D)
	FeynWannParams fwp(&inputMap);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("nBlocks = %d\n", nBlocks);
	logPrintf("Tmin = %lg\n", Tmin);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("Tcount = %lu\n", Tcount);
	logPrintf("neglectThreshold = %lg\n", neglectThreshold);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %lu\n", dmuCount);
	logPrintf("omegaPhByTmin = %lg\n", omegaPhByTmin);
	logPrintf("nModesOverride = %d\n", nModesOverride);
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.needSymmetries = true;
	fwp.needPhonons = true;
	fwp.needSpin = true;
	FeynWann fw(fwp);

	//Determine whether to compute valley contributions (enable if hexagonal lattice):
	double cosTheta12 = dot(fw.R.column(0),fw.R.column(1))/(fw.R.column(0).length()*fw.R.column(1).length());
	double cosTheta13 = dot(fw.R.column(0),fw.R.column(2))/(fw.R.column(0).length()*fw.R.column(2).length());
	double cosTheta23 = dot(fw.R.column(1),fw.R.column(2))/(fw.R.column(1).length()*fw.R.column(2).length());
	bool valley = (fabs(cosTheta12+0.5)<symmThreshold) //120 degrees
		and (fabs(cosTheta13)<symmThreshold) //90 degrees
		and (fabs(cosTheta23)<symmThreshold); //90 degrees
	if(valley) logPrintf("Enabling additional inter-valley weighted calculation for hexagonal lattice.\n\n");
	
	//T array:
	std::vector<double> T(Tcount, Tmin); //set first value here
	for(size_t iT=1; iT<Tcount; iT++) //set remaining values (if any)
		T[iT] = Tmin + iT*(Tmax-Tmin)/(Tcount-1);
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(size_t iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	int nModes = nModesOverride ? std::min(nModesOverride, fw.nModes) : fw.nModes;
	
	//Initialize sampling parameters:
	int nOffsetsPerBlock = ceildiv(nOffsets, nBlocks);
	size_t nKtotPerBlock = fw.eCountPerOffset() * nOffsetsPerBlock;
	size_t nKpairsPerBlock = fw.ePhCountPerOffset() * nOffsetsPerBlock;
	logPrintf("Effectively sampled nKpairs: %lu\n", nKpairsPerBlock * nBlocks);
	if(mpiWorld->isHead())
		logPrintf("%d phonon q-mesh offsets per block parallelized over %d process groups.\n",
			nOffsetsPerBlock, mpiGroupHead->nProcesses());
	
	int oStart = 0, oStop = 0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsetsPerBlock, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets (per block) handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");

	//Determine relevant energy range (states close enough to mu or band edges to matter):
	logPrintf("Determining active energy: "); logFlush();
	EnergyRangeCollect erc(dmu);
	for(int o=oStart; o<oStop; o++)
	{	Random::seed(o);
		vector3<> k0 = fw.randomVector(mpiGroup);
		fw.eLoop(k0, EnergyRangeCollect::eProcess, &erc);
		fw.phLoop(k0, EnergyRangeCollect::phProcess, &erc);
	}
	mpiWorld->allReduce(erc.EvMax, MPIUtil::ReduceMax);
	mpiWorld->allReduce(erc.EcMin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(erc.omegaPhMax, MPIUtil::ReduceMax);
	//--- add margins of max phonon energy, energy conservation width and fermiPrime width
	const double nExponentials = -log(neglectThreshold);
	const double nGaussSigmas = sqrt(-2.*log(neglectThreshold));
	double Emargin = erc.omegaPhMax + nGaussSigmas*EconserveWidth + nExponentials*T.back();
	fw.ePhEstart = erc.EvMax - Emargin;
	fw.ePhEstop = erc.EcMin + Emargin;
	logPrintf("%lg to %lg eV.\n\n", fw.ePhEstart/eV, fw.ePhEstop/eV);
	
	//Collect integrals involved in T1 calculation:
	std::vector<std::shared_ptr<SpinRelaxCollect>> srcArr(nBlocks);
	for(int block=0; block<nBlocks; block++)
	{	logPrintf("Working on block %d of %d: ", block+1, nBlocks); logFlush();
		srcArr[block] = std::make_shared<SpinRelaxCollect>(dmu, T, omegaPhByTmin, nModes, fwp.degeneracyThreshold,
			EconserveWidth, nKpairsPerBlock, nKtotPerBlock, fw.ePhEstart, fw.ePhEstop, valley, fw.R);
		SpinRelaxCollect& src = *(srcArr[block]);
		for(int o=0; o<noMine; o++)
		{	Random::seed(block*nOffsetsPerBlock+o+oStart); //to make results independent of MPI division
			//Process with a random offset pair:
			vector3<> k01 = fw.randomVector(mpiGroup); //must be constant across group
			vector3<> k02 = fw.randomVector(mpiGroup); //must be constant across group
			fw.ePhLoop(k01, k02, SpinRelaxCollect::ePhProcess, &src, //for Gamma
				SpinRelaxCollect::eProcess,   //for Chi at k01
				SpinRelaxCollect::eProcess ); //for Chi at k02
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		//Accumulate over MPI:
		mpiWorld->allReduceData(src.Gamma, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.chi, MPIUtil::ReduceSum);
		if(valley) mpiWorld->allReduceData(src.GammaV, MPIUtil::ReduceSum);
		logPrintf("done.\n"); logFlush();
	}
	
	//Report results with statistics:
	const double ps = 1e3*fs; //picosecond
	const char* formulaName[2] = { "FGR", "dRho" };
	for(size_t iT=0; iT<Tcount; iT++)
	for(size_t iMu=0; iMu<dmuCount; iMu++)
	for(size_t iFormula=0; iFormula<2; iFormula++)
	{	size_t iMuT = 2*(iT*dmuCount + iMu)+iFormula; //combined index
		logPrintf("\nResults for T = %lg K and dmu = %lg eV using %s formula:\n", T[iT]/Kelvin, dmu[iMu]/eV, formulaName[iFormula]);
		std::vector<matrix3<>> Gamma(nBlocks), chi(nBlocks), T1bar(nBlocks), T1Vbar(nBlocks);
		std::vector<double> T1(nBlocks), T1V(nBlocks);
		for(int block=0; block<nBlocks; block++)
		{	SpinRelaxCollect& src = *(srcArr[block]);
			fw.symmetrize(src.Gamma[iMuT]);
			fw.symmetrize(src.chi[iMuT]);
			Gamma[block] = src.Gamma[iMuT];
			chi[block] = src.chi[iMuT];
			T1bar[block] = chi[block] * inv(Gamma[block]);
			T1[block] = (1./3)*trace(T1bar[block]);
			if(valley)
			{	fw.symmetrize(src.GammaV[iMuT]);
				T1Vbar[block] = chi[block] * inv(src.GammaV[iMuT]);
				T1V[block] = (1./3)*trace(T1Vbar[block]);
			}
		}
		reportResult(Gamma, "Gamma", 1./(eV*ps), "1/(eV.ps)");
		reportResult(chi, "chi", 1./eV, "1/eV");
		reportResult(T1bar, "T1", ps, "ps", globalLog, true); //tensor version (averaged on inverse)
		reportResult(T1, "T1", ps, "ps", globalLog, true); //scalar version (averaged on inverse)
		logPrintf("\n");
		if(valley)
		{	reportResult(T1Vbar, "T1valley", ps, "ps", globalLog, true); //tensor version (averaged on inverse)
			reportResult(T1V, "T1valley", ps, "ps", globalLog, true); //scalar version (averaged on inverse)
			logPrintf("\n");
		}
	}
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
