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
#include <core/Units.h>
#include <core/Random.h>
#include <algorithm>
#include "SparseMatrix.h"

static const int ngrid = 1000;

struct SpinRelaxCollect{
	FeynWann& fw;
	const std::vector<double>& dmu; //doping levels
	const std::vector<double>& T; //temperatures
	const double omegaPhByTmin; //lower cutoff in phonon frequency (relative to T)
	const int nModes; //number of phonon modes to include in calculation (override if any, applied below already)

	const double EconserveExpFac, sqrtEconserveExpFac; //exponential factor in Gaussian delta for energy conservation
	const double prefacGamma, prefacChi, prefacG; //prefactors for numerator and denominator of T1
	const double Estart, Estop; //energy range close enough to band edges or mu's to be relevant
	const double degeneracyThreshold;

	std::vector<matrix3<>> Gamma, chi, GammaV; //numerator and denominator in T1^-1, for each T and dmu
	std::vector<std::vector<matrix3<>>> Gamma_mode;
	matrix3<> Gvec, GGT;
	vector3<> K, Kp;
	size_t nKpairContrib;
	std::vector<vector3<double>> weighted_sum_bsq, bsqe, weighted_sum_Bin, Bine; // spin mixing, |internal magnetic field|^2
	std::vector<double> sum_dfde, weighted_sum_Bintot, Bintote; // spin mixing, |internal magnetic field|^2
	std::vector<size_t> nstates;

	SpinRelaxCollect(FeynWann& fw, const std::vector<double>& dmu, const std::vector<double>& T,
		const double& omegaPhMax, const double& omegaPhByTmin, const int& nModes,
		const double& EconserveWidth, size_t nK, size_t nKpairs,
		const double& Estart, const double& Estop, const double& degeneracyThreshold)
		: fw(fw), dmu(dmu), T(T),
		omegaPhByTmin(std::max(1e-3, omegaPhByTmin)), nModes(nModes),
		EconserveExpFac(-0.5 / std::pow(EconserveWidth, 2)), sqrtEconserveExpFac(-0.25 / std::pow(EconserveWidth, 2)),
		prefacChi(0.5 / nK), prefacGamma(2 * M_PI / nKpairs), prefacG(1. / sqrt(sqrt(2.*M_PI)*EconserveWidth)), //include prefactor of Gaussian energy conservation
		Estart(Estart), Estop(Estop),
		Gamma(T.size()*dmu.size()), chi(T.size()*dmu.size()),
		Gamma_mode(T.size()*dmu.size(), std::vector<matrix3<>>(nModes)),
		GammaV(T.size()*dmu.size()), K(1. / 3, 1. / 3, 0), Kp(-1. / 3, -1. / 3, 0),
		degeneracyThreshold(degeneracyThreshold),
		weighted_sum_bsq(T.size()*dmu.size()), sum_dfde(T.size()*dmu.size()), bsqe(ngrid + 1), nstates(ngrid + 1),
		weighted_sum_Bin(T.size()*dmu.size()), weighted_sum_Bintot(T.size()*dmu.size()), Bine(ngrid + 1), Bintote(ngrid + 1)
	{
		Gvec = (2.*M_PI)*inv(fw.R);
		GGT = Gvec * (~Gvec);
		nKpairContrib = 0;
	}

	static inline vector3<> wrap(const vector3<>& x)
	{	vector3<> result = x;
		for (int dir = 0; dir<3; dir++)
			result[dir] -= floor(0.5 + result[dir]);
		return result;
	}
	inline bool isKvalley(vector3<> k) const
	{	return GGT.metric_length_squared(wrap(K - k))
			< GGT.metric_length_squared(wrap(Kp - k));
	}

	void bRange(const FeynWann::StateE& stateE, const double& Estart, const double& Estop, int& bStart, int& bStop){
		bStart = fw.nBands, bStop = 0;
		for (int b = 0; b < fw.nBands; b++){
			const double& E = stateE.E[b];
			if (E >= Estart and b < bStart) bStart = b;
			if (E <= Estop and b >= bStop) bStop = b + 1;
		}
	}

	inline matrix dRho_H1(const diagMatrix& E, const diagMatrix& F, const double& T, const matrix& H1, const int& nBandsSel){
		matrix result(nBandsSel, nBandsSel); complex *rData = result.data();
		double invT = 1. / T;
		for (int b2 = 0; b2 < nBandsSel; b2++)
		for (int b1 = 0; b1 < nBandsSel; b1++){
			if (fabs(E[b1] - E[b2]) <= degeneracyThreshold){
				double Favg = 0.5 * (F[b1] + F[b2]);
				*rData = Favg * (Favg - 1.) * invT * H1(b1, b2);
			}else{
				*rData = (F[b1] - F[b2]) / (E[b1] - E[b2]) * H1(b1, b2);
			}
			rData++;
		}
		return result;
	}

	inline SparseMatrix degenerateProject(const matrix& M, const diagMatrix& E, int bStart, int bStop){
		SparseMatrix result; //nNZ estimate based on diagonal (Rashba)
		for (int b2 = bStart; b2<bStop; b2++){
			const complex* Mdata = M.data() + (b2*M.nRows() + bStart);
			for (int b1 = bStart; b1<bStop; b1++){
				if (fabs(E[b1] - E[b2]) < degeneracyThreshold){
					SparseEntry sr;
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

	void computeChi(const FeynWann::StateE& state){
		//Select relevant band range:
		int bStart, bStop;
		bRange(state, Estart, Estop, bStart, bStop);
		int nBandsSel = bStop - bStart; //reduced number of selected bands at this k-pair
		if (nBandsSel <= 0) return;
		diagMatrix E = state.E(bStart, bStop);

		std::vector<matrix> S(3);
		SparseMatrix Sdeg;
		std::vector<vector3<>> bsq(nBandsSel);
		std::vector<vector3<>> Bin(nBandsSel);
		for (int iDir = 0; iDir < 3; iDir++){
			S[iDir] = state.S[iDir](bStart, bStop, bStart, bStop);
			Sdeg = degenerateProject(S[iDir], E, 0, nBandsSel);
			diagMatrix SiSj = diagSS(Sdeg, Sdeg, nBandsSel);
			for (int b = 0; b < nBandsSel; b++){
				bsq[b][iDir] = 0.5 * (1 - sqrt(SiSj[b]));
				if ((b + bStart) % 2 == 0 && b < nBandsSel - 1)
					Bin[b][iDir] = SiSj[b] * (E[b + 1] - E[b]) * (E[b + 1] - E[b]);
				if ((b + bStart) % 2 == 1 && b > 0)
					Bin[b][iDir] = SiSj[b] * (E[b] - E[b - 1]) * (E[b] - E[b - 1]);
			}
		}

		//Collect results for various dmu values:
		double dE = (Estop - Estart) / ngrid; int iMuMid = (dmu.size() - 1) / 2;
		for (size_t iT = 0; iT < T.size(); iT++){
			double invT = 1. / T[iT];
			for (size_t iMu = 0; iMu < dmu.size(); iMu++){
				size_t iMuT = iT*dmu.size() + iMu; //combined index

				//Compute Fermi occupations and accumulate chi contributions:
				diagMatrix F(nBandsSel);
				for (int b = 0; b < nBandsSel; b++){
					F[b] = fermi(invT*(E[b] - dmu[iMu]));
					double FFbar = F[b] * (1 - F[b]);
					weighted_sum_bsq[iMuT] += FFbar * bsq[b];
					weighted_sum_Bin[iMuT] += FFbar * Bin[b];
					double Bintot = Bin[b][0] + Bin[b][1] + Bin[b][2];
					weighted_sum_Bintot[iMuT] += FFbar * Bintot;
					sum_dfde[iMuT] += FFbar;
					if (iT == T.size() - 1 && iMu == iMuMid){
						int ie = round((E[b] - Estart) / dE);
						if (ie >= 0 && ie <= ngrid){
							bsqe[ie] += bsq[b];
							Bine[ie] += Bin[b]; Bintote[ie] += Bintot;
							nstates[ie]++;
						}
					}
				}
				std::vector<matrix> dRho(3);
				for (int jDir = 0; jDir < 3; jDir++)
					dRho[jDir] = dRho_H1(E, F, T[iT], S[jDir], nBandsSel);

				matrix3<> contrbChi;
				for (int iDir = 0; iDir<3; iDir++)
				for (int jDir = 0; jDir<3; jDir++)
				for (int b2 = 0; b2 < nBandsSel; b2++)
				for (int b1 = 0; b1 < nBandsSel; b1++)
					contrbChi(iDir, jDir) += (S[iDir](b1, b2).conj() * dRho[jDir](b1, b2)).real();
				chi[iMuT] += prefacChi * contrbChi;
			}
		}
	}
	static void eProcess(const FeynWann::StateE& state, void* params){
		((SpinRelaxCollect*)params)->computeChi(state);
	}

	inline bool process(const FeynWann::MatrixEph& mEph){
		const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		const int nBands = e1.E.nRows();

		//Select relevant band range:
		int bStart1, bStop1, bStart2, bStop2;
		bRange(e1, Estart, Estop, bStart1, bStop1);
		bRange(e2, Estart, Estop, bStart2, bStop2);
		int nBandsSel1 = bStop1 - bStart1, nBandsSel2 = bStop2 - bStart2;
		int nBandsSelSq = nBandsSel1 * nBandsSel2;
		if (nBandsSel1 <= 0 || nBandsSel2 <= 0) return false;
		nKpairContrib++;

		double wValley = (isKvalley(e1.k) xor isKvalley(e2.k)) ? 1. : 0.;
		
		diagMatrix E1 = e1.E(bStart1, bStop1), E2 = e2.E(bStart2, bStop2);
		vector3<matrix> S1, S2;
		for (int iDir = 0; iDir<3; iDir++){
			S1[iDir] = e1.S[iDir](bStart1, bStop1, bStart1, bStop1);
			S2[iDir] = e2.S[iDir](bStart2, bStop2, bStart2, bStop2);
		}
		
		//Compute G and [s,G]
		std::vector<matrix> G(nModes, matrix(nBandsSel1, nBandsSel2));
		std::vector<std::vector<matrix>> SGcomm(3, std::vector<matrix>(nModes, matrix(nBandsSel1, nBandsSel2)));
		bool contrib = false;
		std::vector<bool> contribMode(nModes);
		for (int alpha = 0; alpha < nModes; alpha++)
		{	//Phonon occupation (nPh/T and prefactors) for each T:
			contribMode[alpha] = false;
			const double& omegaPh = ph.omega[alpha];
			if (omegaPh / T[0] < omegaPhByTmin) continue; //avoid 0./0. below
			//Energy conservation factor by band pairs:
			std::vector<double> Econserve(nBandsSelSq);
			int bIndex = 0;
			for (int b2 = bStart2; b2 < bStop2; b2++)
			for (int b1 = bStart1; b1 < bStop1; b1++){
				double expTerm = sqrtEconserveExpFac * std::pow(e1.E[b1] - e2.E[b2] - omegaPh, 2);
				//double expTerm = EconserveExpFac * std::pow(e1.E[b1] - e2.E[b2] - omegaPh, 2);
				if (expTerm > -8){ // E1 should > E2, but if adding this restriction is numerically better?
					Econserve[bIndex] = prefacG * exp(expTerm); //compute exponential only when needed
					contribMode[alpha] = true;
				}
				bIndex++;
			}
			// although formula here is not |[s,g]|^2-type any more, 
			if (not contribMode[alpha]) continue; //no energy conserving combination for this phonon mode at present k-pair
			contrib = true;
			//Calculate commutators:
			G[alpha] = mEph.M[alpha](bStart1, bStop1, bStart2, bStop2); //work only with sub-matrix of relevant bands
			complex *GData = G[alpha].data();
			for (int bIndex = 0; bIndex < nBandsSelSq; bIndex++)
				*(GData++) *= Econserve[bIndex];
			// note that even for inner sum, we restrict bStart <= band < bStop
			// this is probably fine suppose bands far from band edges are not involved
			// actually using full bands for inner sum is not safe, since states close or beyond innerWindow are incorrect
			for (int iDir = 0; iDir < 3; iDir++)
				SGcomm[iDir][alpha] = S1[iDir] * G[alpha] - G[alpha] * S2[iDir];
		}

		if (not contrib) return true;
		//Collect results for various dmu values:
		for (size_t iT = 0; iT < T.size(); iT++){
			double invT = 1. / T[iT];
			for (size_t iMu = 0; iMu < dmu.size(); iMu++){
				size_t iMuT = iT*dmu.size() + iMu; //combined index

				diagMatrix F1(nBandsSel1), F2(nBandsSel2);
				for (int b1 = 0; b1 < nBandsSel1; b1++)
					F1[b1] = fermi(invT*(E1[b1] - dmu[iMu]));
				for (int b2 = 0; b2 < nBandsSel2; b2++)
					F2[b2] = fermi(invT*(E2[b2] - dmu[iMu]));
				vector3<matrix> dRho1, dRho2;
				for (int jDir = 0; jDir < 3; jDir++){
					dRho1[jDir] = dRho_H1(E1, F1, T[iT], S1[jDir], nBandsSel1);
					dRho2[jDir] = dRho_H1(E2, F2, T[iT], S2[jDir], nBandsSel2);
				}
				//logPrintf("debug 4\n"); logFlush();
				
				//Accumulate Gamma contributions:
				matrix3<> contribGamma;
				for (int alpha = 0; alpha < nModes; alpha++){
					if (not contribMode[alpha]) continue;
					if (ph.omega[alpha] / T[0] < omegaPhByTmin) continue; //avoid 0./0. below
					double nq = bose(ph.omega[alpha] * invT);

					diagMatrix nF1(nBandsSel1), nF2bar(nBandsSel2);
					for (int b1 = 0; b1 < nBandsSel1; b1++)
						nF1[b1] = nq +  F1[b1];
					for (int b2 = 0; b2 < nBandsSel2; b2++)
						nF2bar[b2] = nq + 1. - F2[b2];
					vector3<matrix> dRhoGcomm;
					for (int jDir = 0; jDir < 3; jDir++)
						dRhoGcomm[jDir] = dRho1[jDir] * G[alpha] * nF2bar - nF1 * G[alpha] * dRho2[jDir];
					
					matrix3<> contribGammaMode;
					for (int jDir = 0; jDir < 3; jDir++)
					for (int iDir = 0; iDir < 3; iDir++)
					for (int b2 = 0; b2 < nBandsSel2; b2++)
					for (int b1 = 0; b1 < nBandsSel1; b1++)
						contribGammaMode(iDir, jDir) += (SGcomm[iDir][alpha](b1, b2).conj() * dRhoGcomm[jDir](b1, b2)).real();
					contribGammaMode = prefacGamma * contribGammaMode;
					contribGamma += contribGammaMode;
					Gamma_mode[iMuT][alpha] += contribGammaMode;
				}
				Gamma[iMuT] += contribGamma;
				GammaV[iMuT] += contribGamma * wValley;
			}
		}
		return true;
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params){
		((SpinRelaxCollect*)params)->process(mEph);
	}
};

//Helper class for collecting relevant energy range:
struct EnergyRangeCollect{
	FeynWann& fw;
	const double &dmuMin, &dmuMax; //minimum and maximum chemical potentials considered
	double EvMax, EcMin; //VBM and CBM estimates
	double omegaPhMax; //max phonon energy
	int band_skipped;

	EnergyRangeCollect(const std::vector<double>& dmu, FeynWann& fw, const int& band_skipped)
		: dmuMin(dmu.front()), dmuMax(dmu.back()),
		EvMax(-DBL_MAX), EcMin(+DBL_MAX), omegaPhMax(0.), fw(fw), band_skipped(band_skipped)
	{}

	void process(const FeynWann::StateE& state){
		if (band_skipped < 0){
			for (const double& E : state.E){
				if (E < 1e-4 and E > EvMax) EvMax = E;
				if (E > 1e-4 and E < EcMin) EcMin = E;
			}
		}
		else{
			if (state.E[fw.nElectrons - band_skipped - 1] > EvMax) EvMax = state.E[fw.nElectrons - band_skipped - 1];
			if (state.E[fw.nElectrons - band_skipped] < EcMin) EcMin = state.E[fw.nElectrons - band_skipped];
		}
	}
	static void eProcess(const FeynWann::StateE& state, void* params){
		((EnergyRangeCollect*)params)->process(state);
	}

	static void phProcess(const FeynWann::StatePh& state, void* params){
		double& omegaPhMax = ((EnergyRangeCollect*)params)->omegaPhMax;
		omegaPhMax = std::max(omegaPhMax, state.omega.back());
	}
};

int main(int argc, char** argv)
{
	InitParams ip = FeynWann::initialize(argc, argv, "Electron-phonon scattering contribution to spin relaxation.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	int nOffsets = inputMap.get("nOffsets", 1); assert(nOffsets > 0);
	const int seedStart = inputMap.get("seedStart", 0);
	int nBlocks = inputMap.get("nBlocks", 1); assert(nBlocks > 0);
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const double Tmin = inputMap.get("Tmin") * Kelvin; //temperature; start of range
	const double Tmax = inputMap.get("Tmax", Tmin / Kelvin) * Kelvin; assert(Tmax >= Tmin); //temperature; end of range (defaults to Tmin)
	const size_t Tcount = inputMap.get("Tcount", 1); assert(Tcount > 0); //number of temperatures
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", dmuMin / eV) * eV; assert(dmuMax >= dmuMin); //optional shift in chemical potential from neutral value; end of range (defaults to dmuMin)
	const size_t dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount > 0); //number of chemical potential shifts (default 1)
	const double omegaPhByTmin = inputMap.get("omegaPhByTmin", 1e-3); //lower cutoff in phonon frequency (relative to temperature)
	const int nModesOverride = inputMap.get("nModesOverride", 0); //if non-zero, use only these many lowest phonon modes (eg. set to 3 for acoustic only in 3D)
	const double degeneracyThreshold = inputMap.get("degeneracyThreshold", 1e-6);
	const double nkBT = inputMap.get("nkBT", 7.); assert(nkBT > 1);
	const int band_skipped = inputMap.get("band_skipped", -1);
	const bool ePhEnable = inputMap.get("ePhEnable", 1);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("seedStart = %d\n", seedStart);
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
	logPrintf("degeneracyThreshold = %lg\n", degeneracyThreshold);
	logPrintf("nkBT = %lg\n", nkBT);
	logPrintf("band_skipped = %d\n", band_skipped);
	logPrintf("ePhEnable = %d\n", ePhEnable);

	//Initialize FeynWann:
	FeynWannParams fwp(&inputMap);	fwp.printParams(); // Bext, EzExt and scissor
	fwp.needSymmetries = true;
	fwp.needPhonons = true;
	fwp.needSpin = true;
	fwp.needVelocity = true;
	FeynWann fw(fwp);

	//T array:
	std::vector<double> T(Tcount, Tmin); //set first value here
	for (size_t iT = 1; iT < Tcount; iT++) //set remaining values (if any)
		T[iT] = Tmin + iT*(Tmax - Tmin) / (Tcount - 1);

	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for (size_t iMu = 1; iMu < dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax - dmuMin) / (dmuCount - 1);
	int nModes = nModesOverride ? std::min(nModesOverride, fw.nModes) : fw.nModes;

	//Initialize sampling parameters:
	int nOffsetsPerBlock = ceildiv(nOffsets, nBlocks);
	size_t nKpairsPerBlock = fw.ePhCountPerOffset() * nOffsetsPerBlock;
	size_t nKPerBlock = fw.eCountPerOffset() * nOffsetsPerBlock;
	logPrintf("Effectively sampled nKpairs: %lu\n", nKpairsPerBlock * nBlocks);
	int oStart = 0, oStop = 0;
	if (mpiGroup->isHead())
		TaskDivision(nOffsetsPerBlock, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop - oStart; //number of offsets (per block) handled by current group
	int oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

	matrix3<> Gvec = (2.*M_PI)*inv(fw.R);
	double qabsmax = 0.;
	for (int i = -1; i <= 1; i = i + 2)
	for (int j = -1; j <= 1; j = j + 2)
	for (int k = -1; k <= 1; k = k + 2){
		vector3<> q; q[0] = 0.5*i; q[1] = 0.5*j; q[2] = 0.5*k;
		q = q * ~Gvec;
		if (q.length()>qabsmax) qabsmax = q.length();
	}
	logPrintf("|q|max = %lf\n", qabsmax);

	if (ip.dryRun)
	{
		logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");

	//Determine relevant energy range (states close enough to mu or band edges to matter):
	EnergyRangeCollect erc(dmu, fw, band_skipped);
	fw.eLoop(vector3<>(), EnergyRangeCollect::eProcess, &erc);
	fw.phLoop(vector3<>(), EnergyRangeCollect::phProcess, &erc);
	mpiWorld->allReduce(erc.EvMax, MPIUtil::ReduceMax);
	mpiWorld->allReduce(erc.EcMin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(erc.omegaPhMax, MPIUtil::ReduceMin);
	logPrintf("EvMax = %lg eV\n", erc.EvMax / eV);
	logPrintf("EcMin = %lg eV\n", erc.EcMin / eV);
	//--- add margins of max phonon energy, energy conservation width and fermiPrime width
	double Emargin = std::max(erc.omegaPhMax + 6.*EconserveWidth, nkBT*T.back()); // erc.omegaPhMax + 6.*EconserveWidth + 20.*T.back();
	double Estart, Estop;
	double Emid = (erc.EvMax + erc.EcMin) / 2.;
	if (fw.isMetal){
		Estart = dmuMin - Emargin;
		Estop = dmuMax + Emargin;
	}
	else{
		Estart = dmuMin > Emid ? dmuMin - Emargin : std::min(dmuMin, erc.EvMax) - Emargin;
		Estop = dmuMax < Emid ? dmuMax + Emargin : std::max(dmuMax, erc.EcMin) + Emargin;
	}
	fw.ePhEstart = Estart; fw.ePhEstop = Estop;
	logPrintf("Emargin = %lg eV\n", Emargin / eV);
	logPrintf("Estart = %lg eV\n", Estart / eV);
	logPrintf("Estop = %lg eV\n", Estop / eV);

	//Collect integrals involved in T1 calculation:
	std::vector<std::shared_ptr<SpinRelaxCollect>> srcArr(nBlocks);
	size_t nKpairContrib = 0;
	for (int block = 0; block < nBlocks; block++){
		logPrintf("Working on block %d of %d: ", block + 1, nBlocks); logFlush();
		srcArr[block] = std::make_shared<SpinRelaxCollect>(fw, dmu, T,
			erc.omegaPhMax, omegaPhByTmin, nModes, EconserveWidth, nKPerBlock, nKpairsPerBlock,
			Estart, Estop, degeneracyThreshold);
		SpinRelaxCollect& src = *(srcArr[block]);
		for (int o = 0; o < noMine; o++)		{
			Random::seed(seedStart + block*nOffsetsPerBlock + o + oStart); //to make results independent of MPI division
			//Process with a random offset pair:
			vector3<> k01 = fw.randomVector(mpiGroup); //must be constant across group
			vector3<> k02 = fw.randomVector(mpiGroup); //must be constant across group
			for (int idir = 0; idir < 3; idir++){
				if (fw.isTruncated[idir]){
					k01[idir] = 0.;
					k02[idir] = 0.;
				}
			}
			fw.eLoop(k01, SpinRelaxCollect::eProcess, &src);
			fw.eLoop(k02, SpinRelaxCollect::eProcess, &src);
			if (ePhEnable) fw.ePhLoop(k01, k02, SpinRelaxCollect::ePhProcess, &src);
			//Print progress:
			if ((o + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o + 1)*100. / noMine))); logFlush(); }
		}
		//Accumulate over MPI:
		mpiWorld->allReduceData(src.Gamma, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.GammaV, MPIUtil::ReduceSum);
		for (std::vector<matrix3<>>& v : src.Gamma_mode)
			mpiWorld->allReduceData(v, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.chi, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.weighted_sum_bsq, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.weighted_sum_Bin, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.weighted_sum_Bintot, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.sum_dfde, MPIUtil::ReduceSum);
		mpiWorld->reduce(src.nKpairContrib, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.bsqe, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.Bine, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.Bintote, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.nstates, MPIUtil::ReduceSum);
		nKpairContrib += src.nKpairContrib;
		logPrintf("done.\n"); logFlush();
	}
	logPrintf("nKpairs contributing to Gamma: %lu (%lu)\n", nKpairContrib, nKpairsPerBlock * nBlocks); logFlush();

	//Report results with statistics:
	if (mpiWorld->isHead()){
		string fnameT1 = "T1_avgGamma.dat"; FILE *fpT1; if (ePhEnable) fpT1 = fopen(fnameT1.c_str(), "w");
		string fnamechi = "chi.dat"; FILE *fpchi; if (ePhEnable) fpchi = fopen(fnamechi.c_str(), "w");
		string fnameGamma = "Gamma.dat"; FILE *fpGamma; if (ePhEnable) fpGamma = fopen(fnameGamma.c_str(), "w");
		string fnamev = "T1_intervalley.dat"; FILE *fpv; if (ePhEnable) fpv = fopen(fnamev.c_str(), "w");
		string fnameintra = "T1_intravalley.dat"; FILE *fpintra; if (ePhEnable) fpintra = fopen(fnameintra.c_str(), "w");
		string fnamemode = "T1_mode.dat"; FILE *fpmode; if (ePhEnable) fpmode = fopen(fnamemode.c_str(), "w");
		if (ePhEnable) { fprintf(fpmode, "Ratio of Rate along x y z of each mode\n"); fflush(fpmode); }
		const char* fnamebsq = "bSq.out"; FILE *fpbsq = fopen(fnamebsq, "w");
		const char* fnamebsqe = "bSqE.out"; FILE *fpbsqe = fopen(fnamebsqe, "w");
		fprintf(fpbsq, "T (K) dmu (eV) bSq_x bSq_y bSq_z\n"); fflush(fpbsq);
		fprintf(fpbsqe, "E (eV) bSqE_x bSqE_y bSqE_z nstates\n"); fflush(fpbsqe);
		const char* fnameBin = "BinterSq.out"; FILE *fpBin = fopen(fnameBin, "w");
		const char* fnameBine = "BinterSqE.out"; FILE *fpBine = fopen(fnameBine, "w");
		fprintf(fpBin, "T (K) dmu (eV) BinSq_x BinSq_y BinSq_z BinSq_tot\n"); fflush(fpBin);
		fprintf(fpBine, "E (eV) BinSqE_x BinSqE_y BinSqE_z BinSqE_tot\n"); fflush(fpBine);
		const double ps = 1e3*fs; //picosecond
		for (size_t iT = 0; iT < Tcount; iT++)
		for (size_t iMu = 0; iMu < dmuCount; iMu++)
		{
			size_t iMuT = iT*dmuCount + iMu; //combined index
			logPrintf("\nResults for T = %lg K and dmu = %lg eV:\n", T[iT] / Kelvin, dmu[iMu] / eV);
			std::vector<matrix3<>> Gamma(nBlocks), Gammaavg(1), Gammasym(nBlocks), chi(1), chisym(1), T1bar(nBlocks), T1barsym(nBlocks), T1barVsym(nBlocks), T1barIntrasym(nBlocks);
			std::vector<double> T1(nBlocks), T1sym(nBlocks);
			std::vector<matrix3<>> Gammaavg_mode(nModes);
			vector3<> weighted_sum_bsq_avg, bsq_avg, weighted_sum_Bin_avg, Bin_avg;
			std::vector<vector3<>> bsqe(ngrid + 1), Bine(ngrid + 1); std::vector<double> nstates(ngrid + 1), Bintote(ngrid + 1);
			double sum_dfde_avg = 0, weighted_sum_Bintot_avg = 0, Bintot_avg = 0;
			// Neglect the variation of chi
			for (int block = 0; block < nBlocks; block++){
				SpinRelaxCollect& src = *(srcArr[block]);
				chi[0] += src.chi[iMuT] / nBlocks;
				fw.symmetrize(src.chi[iMuT]);
				chisym[0] += src.chi[iMuT] / nBlocks;
				weighted_sum_bsq_avg += src.weighted_sum_bsq[iMuT] / nBlocks;
				weighted_sum_Bin_avg += src.weighted_sum_Bin[iMuT] / nBlocks;
				weighted_sum_Bintot_avg += src.weighted_sum_Bintot[iMuT] / nBlocks;
				sum_dfde_avg += src.sum_dfde[iMuT] / nBlocks;
				for (int ie = 0; ie < ngrid + 1; ie++){
					bsqe[ie] += src.bsqe[ie] / nBlocks;
					Bine[ie] += src.Bine[ie] / nBlocks; Bintote[ie] += src.Bintote[ie] / nBlocks;
					nstates[ie] += src.nstates[ie] / nBlocks;
				}
			}
			matrix3<> invchi = inv(chi[0]);
			matrix3<> invchisym = inv(chisym[0]);

			for (int block = 0; block < nBlocks; block++){
				SpinRelaxCollect& src = *(srcArr[block]);
				Gamma[block] = src.Gamma[iMuT];
				Gammaavg[0] += Gamma[block] / nBlocks;
				T1bar[block] = chi[0] * inv(src.Gamma[iMuT]);
				fw.symmetrize(src.Gamma[iMuT]);
				Gammasym[block] = src.Gamma[iMuT];
				T1barsym[block] = chisym[0] * inv(src.Gamma[iMuT]);
				fw.symmetrize(src.GammaV[iMuT]);
				T1barVsym[block] = chi[0] * inv(src.GammaV[iMuT]);
				T1barIntrasym[block] = chi[0] * inv(src.Gamma[iMuT] - src.GammaV[iMuT]);
				for (int alpha = 0; alpha < nModes; alpha++)
					Gammaavg_mode[alpha] += src.Gamma_mode[iMuT][alpha] / nBlocks;
			}

			bsq_avg = weighted_sum_bsq_avg / sum_dfde_avg;
			fprintf(fpbsq, "%lg %lg %lg %lg %lg\n", T[iT] / Kelvin, dmu[iMu] / eV, bsq_avg[0], bsq_avg[1], bsq_avg[2]); fflush(fpbsq);
			if (iT == Tcount - 1 && iMu == (dmuCount - 1) / 2){
				double dE = (Estop - Estart) / ngrid;
				for (int ie = 0; ie < ngrid + 1; ie++){
					if (nstates[ie] > 0){
						bsqe[ie] /= nstates[ie];
						fprintf(fpbsqe, "%lg %lg %lg %lg %lg\n", (Estart + dE * ie) / eV, bsqe[ie][0], bsqe[ie][1], bsqe[ie][2], nstates[ie]); fflush(fpbsqe);
					}
				}
			}
			Bin_avg = weighted_sum_Bin_avg / sum_dfde_avg; Bintot_avg = weighted_sum_Bintot_avg / sum_dfde_avg;
			fprintf(fpBin, "%lg %lg %lg %lg %lg %lg\n", T[iT] / Kelvin, dmu[iMu] / eV, Bin_avg[0], Bin_avg[1], Bin_avg[2], Bintot_avg); fflush(fpBin);
			if (iT == Tcount - 1 && iMu == (dmuCount - 1) / 2){
				double dE = (Estop - Estart) / ngrid;
				for (int ie = 0; ie < ngrid + 1; ie++){
					if (nstates[ie] > 0){
						Bine[ie] /= nstates[ie]; Bintote[ie] /= nstates[ie];
						fprintf(fpBine, "%lg %lg %lg %lg %lg\n", (Estart + dE * ie) / eV, Bine[ie][0], Bine[ie][1], Bine[ie][2], Bintote[ie]); fflush(fpBine);
					}
				}
			}

			if (ePhEnable){
				reportResult(T1barsym, "T1sym", ps, "ps", stdout, true); //tensor version

				for (int ii = 0; ii < 3; ii++)
				for (int jj = 0; jj < 3; jj++){
					fprintf(fpchi, "%lg ", chi[0](ii, jj)*eV);
					fprintf(fpGamma, "%lg ", Gammaavg[0](ii, jj)*eV*ps);
				}
				fprintf(fpchi, "\n");	fprintf(fpGamma, "\n");

				fprintf(fpT1, "T = %lg K and dmu = %lg eV:\n", T[iT] / Kelvin, dmu[iMu] / eV);
				reportResult(T1bar, "T1", ps, "ps", fpT1, true); //tensor version

				fprintf(fpv, "T = %lg K and dmu = %lg eV:\n", T[iT] / Kelvin, dmu[iMu] / eV);
				reportResult(T1barVsym, "T1Vsym", ps, "ps", fpv, true); //tensor version
				fprintf(fpintra, "T = %lg K and dmu = %lg eV:\n", T[iT] / Kelvin, dmu[iMu] / eV);
				reportResult(T1barIntrasym, "T1Intrasym", ps, "ps", fpintra, true); //tensor version

				fprintf(fpmode, "Results for T = %lg K and dmu = %lg eV:\n", T[iT] / Kelvin, dmu[iMu] / eV);
				for (int alpha = 0; alpha < nModes; alpha++){
					fprintf(fpmode, "mode %d: %lg %lg %lg", alpha,
						Gammaavg_mode[alpha](0, 0) / Gammaavg[0](0, 0), Gammaavg_mode[alpha](1, 1) / Gammaavg[0](1, 1), Gammaavg_mode[alpha](2, 2) / Gammaavg[0](2, 2));
					fprintf(fpmode, "\n"); fflush(fpmode);
				}
			}
		}
		if (ePhEnable){
			fclose(fpT1); fclose(fpchi); fclose(fpGamma); fclose(fpv); fclose(fpintra); fclose(fpmode);
		}
		fclose(fpbsq); fclose(fpbsqe); fclose(fpBin); fclose(fpBine);
	}

	fw.free();
	FeynWann::finalize();
	return 0;
}
