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
#include <algorithm>

struct SpinRelaxCollect
{	const std::vector<double>& dmu; //doping levels
	const std::vector<double>& T; //temperatures
	const double omegaPhByTmin; //lower cutoff in phonon frequency (relative to T)
	const int nModes; //number of phonon modes to include in calculation (override if any, applied below already)
	
	const double EconserveExpFac; //exponential factor in Gaussian delta for energy conservation
	const double prefacGamma, prefacChi; //prefactors for numerator and denominator of T1
	const double Estart, Estop; //energy range close enough to band edges or mu's to be relevant
	std::vector<matrix3<>> Gamma, chi; //numerator and denominator in T1^-1, for each T and dmu 
	
	//HACK for valley contrib
	const matrix3<> G; 
	matrix3<> GGT;
	vector3<> K, Kp;
	
	SpinRelaxCollect(const std::vector<double>& dmu, const std::vector<double>& T, double omegaPhByTmin, int nModes,
		double EconserveWidth, size_t nKpairs, double Estart, double Estop, matrix3<> G)
	: dmu(dmu), T(T), omegaPhByTmin(std::max(1e-3,omegaPhByTmin)), nModes(nModes),
		EconserveExpFac(-0.5/std::pow(EconserveWidth, 2)),
		prefacGamma(2*M_PI/ (nKpairs * sqrt(2.*M_PI)*EconserveWidth)), //include prefactor of Gaussian energy conservation
		prefacChi(0.5/nKpairs), //collected over both k in each k-pair for consistency
		Estart(Estart), Estop(Estop), G(G),
		Gamma(T.size()*dmu.size()), chi(T.size()*dmu.size()),
		K(1./3, 1./3, 0), //HACK for valley contrib
		Kp(-1./3, -1./3, 0) //HACK for valley contrib
	{
		GGT = G * (~G); //HACK for valley contrib
	}
	
	inline SparseMatrix degenerateProject(const matrix& M, const diagMatrix& E, int bStart, int bStop)
	{	static const double degeneracyThreshold = 1e-6;
		SparseMatrix result; result.reserve(bStop-bStart); //typically diagonal (Rashba) or block diagonal with size 2 (Kramer-degenerate)
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
	//HACK block valley contrib
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
	
	void process(const FeynWann::MatrixEph& mEph)
	{	const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		const int nBands = e1.E.nRows();
		
		//Select relevant band range:
		int bStart = nBands, bStop = 0;
		#define SET_bRange(s) \
			for(int b=0; b<nBands; b++) \
			{	const double& E = e##s.E[b]; \
				if(E>=Estart and b<bStart) bStart=b; \
				if(E<=Estop and b>=bStop) bStop=b+1; \
			}
		SET_bRange(1)
		SET_bRange(2)
		#undef SET_bRange
		int nBandsSel = bStop - bStart; //reduced number of selected bands at this k-pair
		int nBandsSelSq = nBandsSel * nBandsSel;
		if(nBandsSel <= 0) return;
		
		//Degenerate spin projections:
		std::vector<SparseMatrix> Sdeg1(3), Sdeg2(3);
		for(int iDir=0; iDir<3; iDir++)
		{	Sdeg1[iDir] = degenerateProject(e1.S[iDir], e1.E, bStart, bStop);
			Sdeg2[iDir] = degenerateProject(e2.S[iDir], e2.E, bStart, bStop);
		}
		
		//Compute chi contributions by band except for electron occupation factors:
		#define CONTRIB_chi(s) \
			std::vector<matrix3<>> contribChi##s(nBands); \
			for(int iDir=0; iDir<3; iDir++) \
			for(int jDir=0; jDir<3; jDir++) \
			{	diagMatrix SiSj = diagSS(Sdeg##s[iDir], Sdeg##s[jDir], nBandsSel); \
				for(int b=bStart; b<bStop; b++) \
					contribChi##s[b](iDir,jDir) = prefacChi * SiSj[b-bStart]; \
			}
		CONTRIB_chi(1)
		CONTRIB_chi(2)
		#undef CONTRIB_chi
		
		//HACK block for valley contrib
		bool isK1 = isKvalley(e1.k);
		bool isK2 = isKvalley(e2.k);
		double wValley = (isK1 xor isK2) ? 1. : 0.;
		
		//Compute Gamma contributions by band pair and T, except for electron occupation factors:
		std::vector<std::vector<matrix3<>>> contribGamma(T.size(), std::vector<matrix3<>>(nBandsSelSq));
		for(int alpha=0; alpha<nModes; alpha++)
		{	//Phonon occupation (nPh/T and prefactors) for each T:
			const double& omegaPh = ph.omega[alpha];
			std::vector<double> prefac_nPhByT(T.size());
			for(size_t iT=0; iT<T.size(); iT++)
			{	double invT = 1./T[iT];
				const double omegaPhByT = invT*omegaPh;
				if(omegaPhByT < omegaPhByTmin) continue; //avoid 0./0. below
				prefac_nPhByT[iT] =  prefacGamma * invT * bose(omegaPhByT);
			}
			//Energy conservation factor by band pairs:
			std::vector<double> Econserve(nBandsSelSq);
			int bIndex = 0;
			bool contrib = false;
			for(int b2=bStart; b2<bStop; b2++)
			for(int b1=bStart; b1<bStop; b1++)
			{	double expTerm = EconserveExpFac * std::pow(e1.E[b1] - e2.E[b2] - omegaPh, 2);
				if(expTerm > -15.)
				{	Econserve[bIndex] = exp(expTerm); //compute exponential only when needed
					contrib = true;
				}
				bIndex++;
			}
			if(not contrib) continue; //no energy conserving combination for this phonon mode at present k-pair
			//Calculate commutators:
			const matrix G = mEph.M[alpha](bStart,bStop, bStart,bStop); //work only with sub-matrix of relevant bands
			matrix SGcomm[3]; vector3<const complex*> SGcommData;
			for(int iDir=0; iDir<3; iDir++)
			{	SGcomm[iDir] = Sdeg1[iDir] * G - G * Sdeg2[iDir];
				SGcommData[iDir] = SGcomm[iDir].data();
			}
			//Collect commutator contributions:
			for(int bIndex=0; bIndex<nBandsSelSq; bIndex++) //loop over b2 and b1
			{	vector3<complex> SGcommCur = loadVector(SGcommData, bIndex);
				matrix3<> SGcommOuter = realOuter(SGcommCur, SGcommCur);
				for(size_t iT=0; iT<T.size(); iT++)
				{	contribGamma[iT][bIndex] += (prefac_nPhByT[iT] * Econserve[bIndex]) * SGcommOuter; //* wValley; add for wValley contrib //HACK
				}
			}
		}
		
		//Collect results for various dmu values:
		for(size_t iT=0; iT<T.size(); iT++)
		{	double invT = 1./T[iT];
			for(size_t iMu=0; iMu<dmu.size(); iMu++)
			{	size_t iMuT = iT*dmu.size() + iMu; //combined index
				//Compute Fermi occupations and accumulate chi contributions:
				#define CALC_F_ACCUM_CHI(s) \
					diagMatrix F##s(nBands), Fbar##s(nBands); \
					for(int b=bStart; b<bStop; b++) \
					{	fermi(invT*(e##s.E[b] - dmu[iMu]), F##s[b], Fbar##s[b]); \
						chi[iMuT] += (invT * F##s[b]*Fbar##s[b]) * contribChi##s[b]; \
					}
				CALC_F_ACCUM_CHI(1)
				CALC_F_ACCUM_CHI(2)
				#undef CALC_F_ACCUM_CHI
				
				//Accumulate Gamma contributions:
				int bIndex = 0;
				for(int b2=bStart; b2<bStop; b2++)
				for(int b1=bStart; b1<bStop; b1++)
				{	Gamma[iMuT] += contribGamma[iT][bIndex] * (F2[b2] * Fbar1[b1]);
					bIndex++;
				}
			}
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params)
	{	((SpinRelaxCollect*)params)->process(mEph);
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
	{	for(const double& E: state.E)
		{	if(E<dmuMin and E>EvMax) EvMax = E;
			if(E>dmuMax and E<EcMin) EcMin = E;
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
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const double Tmin = inputMap.get("Tmin") * Kelvin; //temperature; start of range
	const double Tmax = inputMap.get("Tmax", Tmin/Kelvin) * Kelvin; assert(Tmax>=Tmin); //temperature; end of range (defaults to Tmin)
	const size_t Tcount = inputMap.get("Tcount", 1); assert(Tcount>0); //number of temperatures
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", dmuMin/eV) * eV; assert(dmuMax>=dmuMin); //optional shift in chemical potential from neutral value; end of range (defaults to dmuMin)
	const size_t dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts (default 1)
	const double omegaPhByTmin = inputMap.get("omegaPhByTmin", 1e-3); //lower cutoff in phonon frequency (relative to temperature)
	const int nModesOverride = inputMap.get("nModesOverride", 0); //if non-zero, use only these many lowest phonon modes (eg. set to 3 for acoustic only in 3D)
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("nBlocks = %d\n", nBlocks);
	logPrintf("Tmin = %lg\n", Tmin);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("Tcount = %lu\n", Tcount);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %lu\n", dmuCount);
	logPrintf("omegaPhByTmin = %lg\n", omegaPhByTmin);
	logPrintf("nModesOverride = %d\n", nModesOverride);
	
	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needSymmetries = true;
	fwp.needPhonons = true;
	fwp.needSpin = true;
	FeynWann fw(fwp);

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
	size_t nKpairsPerBlock = fw.ePhCountPerOffset() * nOffsetsPerBlock;
	logPrintf("Effectively sampled nKpairs: %lu\n", nKpairsPerBlock * nBlocks);
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
	EnergyRangeCollect erc(dmu);
	fw.eLoop(vector3<>(), EnergyRangeCollect::eProcess, &erc);
	fw.phLoop(vector3<>(), EnergyRangeCollect::phProcess, &erc);
	mpiWorld->allReduce(erc.EvMax, MPIUtil::ReduceMax);
	mpiWorld->allReduce(erc.EcMin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(erc.omegaPhMax, MPIUtil::ReduceMax);
	//--- add margins of max phonon energy, energy conservation width and fermiPrime width
	double Emargin = erc.omegaPhMax + 6.*EconserveWidth + 20.*T.back();
	double Estart = erc.EvMax - Emargin;
	double Estop = erc.EcMin + Emargin;
	matrix3<> G = 2*M_PI * inv(fw.R); //HACK for valley contrib
	//Collect integrals involved in T1 calculation:
	std::vector<std::shared_ptr<SpinRelaxCollect>> srcArr(nBlocks);
	for(int block=0; block<nBlocks; block++)
	{	logPrintf("Working on block %d of %d: ", block+1, nBlocks); logFlush();
		srcArr[block] = std::make_shared<SpinRelaxCollect>(dmu, T, omegaPhByTmin, nModes, EconserveWidth, nKpairsPerBlock, Estart, Estop, G);
		SpinRelaxCollect& src = *(srcArr[block]);
		for(int o=0; o<noMine; o++)
		{	Random::seed(block*nOffsetsPerBlock+o+oStart); //to make results independent of MPI division
			//Process with a random offset pair:
			vector3<> k01 = fw.randomVector(mpiGroup); //must be constant across group
			vector3<> k02 = fw.randomVector(mpiGroup); //must be constant across group
			fw.ePhLoop(k01, k02, SpinRelaxCollect::ePhProcess, &src);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		//Accumulate over MPI:
		mpiWorld->allReduceData(src.Gamma, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.chi, MPIUtil::ReduceSum);
		logPrintf("done.\n"); logFlush();
	}
	
	//Report results with statistics:
	const double ps = 1e3*fs; //picosecond
	for(size_t iT=0; iT<Tcount; iT++)
	for(size_t iMu=0; iMu<dmuCount; iMu++)
	{	size_t iMuT = iT*dmuCount + iMu; //combined index
		logPrintf("\nResults for T = %lg K and dmu = %lg eV:\n", T[iT]/Kelvin, dmu[iMu]/eV);
		std::vector<matrix3<>> Gamma(nBlocks), chi(nBlocks), T1bar(nBlocks);
		std::vector<double> T1(nBlocks);
		for(int block=0; block<nBlocks; block++)
		{	SpinRelaxCollect& src = *(srcArr[block]);
			fw.symmetrize(src.Gamma[iMuT]);
			fw.symmetrize(src.chi[iMuT]);
			Gamma[block] = src.Gamma[iMuT];
			chi[block] = src.chi[iMuT];
			T1bar[block] = chi[block] * inv(Gamma[block]);
			T1[block] = (1./3)*trace(T1bar[block]);
		}
		reportResult(Gamma, "Gamma", 1./(eV*ps), "1/(eV.ps)");
		reportResult(chi, "chi", 1./eV, "1/eV");
		reportResult(T1bar, "T1", ps, "ps"); //tensor version
		reportResult(T1, "T1", ps, "ps"); //tensor version
	}
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
