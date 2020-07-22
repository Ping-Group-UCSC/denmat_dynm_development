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

/*
This program is used to output quanties for denmat_dynm code by JXU:
number of bands; number of valence bands; band index for e-ph
number of k points in the whole grids (needed in prefactor) and in energy windowns;
eigen energies;
spin matrices;

number of k-k' piars in energy windowns;
kpair Information: (i) indeces of k and k' (ii) P1_kk' and P2_kk'

Notice that for semiconductor, conduction and valence electrons are dealt with separately in JXU's code
*/

#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "help_lindbladInit_for-DMD-4.4.h"
#include <gsl/gsl_cblas.h>

const double Tesla = eV*sec / (meter*meter);

static const double omegaPhCut = 1e-6;
static const double nEphDelta_default = 4.; //number of ePhDelta to include in output
static const double degthr = 1e-8;

//Reverse iterator for pointers:
template<class T> constexpr std::reverse_iterator<T*> reverse(T* i) { return std::reverse_iterator<T*>(i); }

//Helper class to "argsort" an array i.e. determine the indices that sort it
template<typename ArrayType> struct IndexCompare
{
	const ArrayType& array;
	IndexCompare(const ArrayType& array) : array(array) {}
	template<typename Integer> bool operator()(Integer i1, Integer i2) const { return array[i1] < array[i2]; }
};

//Lindblad initialization using FeynWann callback
struct LindbladInit
{
	bool DEBUG;
	FeynWann& fw;
	const vector3<int>& NkFine; //!< effective k-point mesh sampled
	const size_t nkTot; //!< total k-points effectively used in BZ sampling

	const double dmuMin, dmuMax, Tmax;
	double nkBT;
	const double pumpOmegaMax, pumpTau, probeOmegaMax;

	const bool ePhEnabled, ePhOnlyElec, ePhOnlyHole; //!< whether e-ph coupling is enabled
	const double ePhDelta; //!< Gaussian energy conservation width
	double nEphDelta;
	bool detailBalance, variedDelta, antiResonant; // techniques to deal with delta functions
	bool printgm, mergegm;
	int band_skipped;
	int modeStart, modeStop;

	// ?Start and ?Stop are used most frequently and used as global variables in some subroutines
	// ?_probe for probe and can contain bands far from band edges; ?_eph for e-ph scattering; ?_dm for states related to density matrix change
	int bStart, bStop, bCBM, nBandsSel, bBot_eph, bTop_eph, bBot_dm, bTop_dm, bBot_probe, bTop_probe, bRef, nBandsSel_probe;
	double EvMax, EcMin, Emid; //VBM and CBM estimates, Emid is the energy in the middle of the gap
	double Estart, Estop, EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph; //energy range for k selection

	LindbladInit(FeynWann& fw, const vector3<int>& NkFine,
		double dmuMin, double dmuMax, double Tmax,
		double pumpOmegaMax, double pumpTau, double probeOmegaMax,
		bool ePhEnabled, bool ePhOnlyElec, bool ePhOnlyHole, double ePhDelta)
		: fw(fw), NkFine(NkFine), nkTot(NkFine[0] * NkFine[1] * NkFine[2]),
		dmuMin(dmuMin), dmuMax(dmuMax), Tmax(Tmax),
		pumpOmegaMax(pumpOmegaMax), pumpTau(pumpTau), probeOmegaMax(probeOmegaMax),
		ePhEnabled(ePhEnabled), ePhOnlyElec(ePhOnlyElec), ePhOnlyHole(ePhOnlyHole), ePhDelta(ePhDelta)
	{
	}

	//--------- Part 1: k-point selection -------------

	inline void eRange(const FeynWann::StateE& state){
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
	static void eRange(const FeynWann::StateE& state, void* params){
		((LindbladInit*)params)->eRange(state);
	}

	inline void bSelect(const FeynWann::StateE& state){
		for (int b = 0; b < fw.nBands; b++){
			const double& E = state.E[b];
			if (E >= Estart and b < bStart) bStart = b;
			if (E <= Estop and b >= bStop) bStop = b + 1;
			if (E <= Emid and b >= bCBM) bCBM = b + 1;
		}
	}
	static void bSelect(const FeynWann::StateE& state, void* params){
		((LindbladInit*)params)->bSelect(state);
	}
	void bSelect_driver(const double& EBot, const double& ETop, int& bBot, int& bTop){
		Estart = EBot; Estop = ETop;
		bStart = fw.nBands; bStop = 0; bCBM = 0;
		for (vector3<> qOff : fw.qOffset)
			fw.eLoop(qOff, LindbladInit::bSelect, this);
		mpiWorld->allReduce(bStart, MPIUtil::ReduceMin);
		mpiWorld->allReduce(bStop, MPIUtil::ReduceMax);
		mpiWorld->allReduce(bCBM, MPIUtil::ReduceMax);
		bBot = bStart; bTop = bStop;
	}

	std::vector<vector3<>> k; //selected k-points
	std::vector<double> E; //all band energies for selected k-points

	std::vector<vector3<int>> offK; //index of each k by offsets: [ kOffset, qOffset, ik ]
	vector3<int> offKcur; //offset indices of current eLoop call in first two components
	std::vector<vector3<int>> offKuniq; //unique offsets in offK
	std::vector<size_t> ikStartOff; //starting k index for each offset in offKuniq (length: offKuniq.size()+1)

	size_t iGroup; //index of current group amongst groups
	size_t nGroups; //number of process groups
	std::vector<size_t> offStartGroup; //starting offset index for each process group (length: nGroups+1)
	std::vector<size_t> ikStartGroup; //starting k index for each process group (length: nGroups+1)

	std::map<size_t, size_t> kIndexMap; //map from k-point mesh index to index in selected set
	inline size_t kIndex(vector3<> k){
		size_t index = 0;
		for (int iDir = 0; iDir < 3; iDir++){
			double ki = k[iDir] - floor(k[iDir]); //wrapped to [0,1)
			index = (size_t)round(NkFine[iDir] * (index + ki));
		}
		return index;
	}
	//Search for k using kIndexMap; return false if not found
	inline bool findK(vector3<> k, size_t&ik)
	{
		const std::map<size_t, size_t>::iterator iter = kIndexMap.find(kIndex(k));
		if (iter != kIndexMap.end())
		{
			ik = iter->second;
			return true;
		}
		else return false;
	}

	inline void kSelect(const FeynWann::StateE& state){
		bool active = false;
		//bool has_elec = false, has_hole = false;
		for (double E : state.E)
		if (E >= Estart and E <= Estop){
			active = true; break;
		}
		if (active){
			k.push_back(state.k);
			E.insert(E.end(), &state.E[bBot_probe], &state.E[bTop_probe]);
			offKcur[2] = state.ik;
			offK.push_back(offKcur);
		}
	}
	static void kSelect(const FeynWann::StateE& state, void* params){
		((LindbladInit*)params)->kSelect(state);
	}
	void kpointSelect(const std::vector<vector3<>>& k0){
		//Parallel:
		size_t oStart, oStop; //range of offstes handled by this process group
		if (mpiGroup->isHead())
			TaskDivision(k0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart); mpiGroup->bcast(oStop);
		size_t noMine = oStop - oStart;
		size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress
		
		// Determin VBM and CBM
		EvMax = -DBL_MAX; EcMin = +DBL_MAX;
		//for (size_t o = oStart; o<oStop; o++){
		//	for (vector3<> qOff : fw.qOffset) fw.eLoop(k0[o] + qOff, LindbladInit::eRange, this);
		//	if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); }
		//}
		for (vector3<> qOff : fw.qOffset) fw.eLoop(qOff, LindbladInit::eRange, this);
		mpiWorld->allReduce(EvMax, MPIUtil::ReduceMax); mpiWorld->allReduce(EcMin, MPIUtil::ReduceMin);
		logPrintf("VBM: %.3lf eV, CBM: %.3lf eV\n", EvMax / eV, EcMin / eV);
		logPrintf("Note that VBM and CBM may not be determined correctly,\nyou may have to use band_skipped to set starting band index of wannier bands\n");
		Emid = (EvMax + EcMin) / 2.;

		//Determine energy range:
		//--- add margins of max phonon energy, energy conservation width and fermiPrime width
		double Emargin = nkBT*Tmax; //neglect below 10^-3 occupation deviation from equilibrium
		if (!fw.isMetal){
			logPrintf("The system is not metalic\n");
			double EBot_pump = EcMin - pumpOmegaMax - 5. / pumpTau;
			if (ePhOnlyElec)
				EBot_eph = Emid;
			else
				EBot_eph = std::min(EBot_pump, std::min(EvMax, dmuMin) - Emargin);
			EBot_dm = std::min(EBot_pump, EBot_eph);

			double ETop_pump = EvMax + pumpOmegaMax + 5. / pumpTau;
			if (ePhOnlyHole)
				ETop_eph = Emid;
			else
				ETop_eph = std::max(ETop_pump, std::max(EcMin, dmuMax) + Emargin);
			ETop_dm = std::max(ETop_pump, ETop_eph);
		}
		else{
			logPrintf("The system is treated as metal\n");
			EBot_eph = dmuMin - pumpOmegaMax - std::max(5. / pumpTau + 3.*Tmax, Emargin);
			EBot_dm = EBot_eph;
			ETop_eph = dmuMax + pumpOmegaMax + std::max(5. / pumpTau + 3.*Tmax, Emargin);
			ETop_dm = ETop_eph;
		}
		EBot_probe = EBot_dm - probeOmegaMax - 5. / pumpTau;
		ETop_probe = ETop_dm + probeOmegaMax + 5. / pumpTau;
		logPrintf("Active energy range for probe: %.3lf to %.3lf eV\n", EBot_probe / eV, ETop_probe / eV);
		logPrintf("Active energy range for kpointSelect: %.3lf to %.3lf eV\n", EBot_eph / eV, ETop_eph / eV);
		logPrintf("Active energy range for density matrix: %.3lf to %.3lf eV\n", EBot_dm / eV, ETop_dm / eV);

		//Select bands:
		bSelect_driver(EBot_probe, ETop_probe, bBot_probe, bTop_probe);
		nBandsSel_probe = bTop_probe - bBot_probe; bRef = bBot_probe;
		logPrintf("\nbBot_probe= %d bTop_probe= %d\n", bBot_probe, bTop_probe);
		bSelect_driver(EBot_dm, ETop_dm, bBot_dm, bTop_dm);
		logPrintf("\nbBot_dm= %d bTop_dm= %d\n", bBot_dm, bTop_dm);
		bSelect_driver(EBot_eph, ETop_eph, bBot_eph, bTop_eph);
		if (fw.isMetal) bCBM = bBot_probe;
		logPrintf("\nbBot_eph= %d bTop_eph= %d bCBM= %d\n", bBot_eph, bTop_eph, bCBM);

		//Select k-points:
		Estart = EBot_eph; Estop = ETop_eph;
		bStart = bBot_eph; bStop = bTop_eph; nBandsSel = bStop - bStart;
		logPrintf("Scanning k-points with active states: "); logFlush();
		for (size_t o = oStart; o<oStop; o++){
			offKcur[0] = o;
			offKcur[1] = 0;
			for (vector3<> qOff : fw.qOffset){
				fw.eLoop(k0[o] + qOff, LindbladInit::kSelect, this);
				offKcur[1]++; //increment qOffset index
			}
			if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();

		//Synchronize selected k and E across all processes:
		//--- determine nk on each process and compute cumulative counts
		std::vector<size_t> nkPrev(mpiWorld->nProcesses() + 1);
		for (int jProc = 0; jProc < mpiWorld->nProcesses(); jProc++){
			size_t nkCur = k.size();
			mpiWorld->bcast(nkCur, jProc); //nkCur = k.size() on jProc in all processes
			nkPrev[jProc + 1] = nkPrev[jProc] + nkCur; //cumulative count
		}
		size_t nkSelected = nkPrev.back();
		//--- broadcast k and E:
		{	//Set k and E in position in global arrays:
			std::vector<vector3<>> k(nkSelected);
			std::vector<double> E(nkSelected*nBandsSel_probe); // notice nBandsSel_probe
			std::vector<vector3<int>> offK(nkSelected);
			std::copy(this->k.begin(), this->k.end(), k.begin() + nkPrev[mpiWorld->iProcess()]);
			std::copy(this->E.begin(), this->E.end(), E.begin() + nkPrev[mpiWorld->iProcess()] * nBandsSel_probe); // notice nBandsSel_probe
			std::copy(this->offK.begin(), this->offK.end(), offK.begin() + nkPrev[mpiWorld->iProcess()]);
			//Broadcast:
			for (int jProc = 0; jProc < mpiWorld->nProcesses(); jProc++){
				size_t ikStart = nkPrev[jProc], nk = nkPrev[jProc + 1] - ikStart;
				mpiWorld->bcast(k.data() + ikStart, nk, jProc);
				mpiWorld->bcast(E.data() + ikStart*nBandsSel_probe, nk*nBandsSel_probe, jProc); // notice nBandsSel_probe
				mpiWorld->bcast(offK.data() + ikStart, nk, jProc);
			}
			//Sort by offset:
			std::vector<size_t> sortIndex(nkSelected);
			for (size_t i = 0; i<nkSelected; i++) sortIndex[i] = i;
			std::sort(sortIndex.begin(), sortIndex.end(), IndexCompare<std::vector<vector3<int>>>(offK));
			//Store to class variables in sorted order:
			this->k.resize(nkSelected);
			this->E.resize(nkSelected*fw.nBands);
			this->offK.resize(nkSelected);
			vector3<int> offKprev(-1, -1, 0);
			for (size_t i = 0; i<nkSelected; i++)
			{
				size_t iSrc = sortIndex[i];
				this->k[i] = k[iSrc];
				eblas_copy(this->E.data() + i*fw.nBands, E.data() + iSrc*fw.nBands, fw.nBands);
				this->offK[i] = offKcur = offK[iSrc];
				//Update unique offset list:
				offKcur[2] = 0; //ignore k in comparing for unique list
				if (not(offKcur == offKprev))
				{
					offKuniq.push_back(offKcur);
					ikStartOff.push_back(i);
					offKprev = offKcur;
				}
			}
			ikStartOff.push_back(nkSelected);
		}
		logPrintf("Found k-points with active states in %lu of %lu q-mesh offsets (%.0fx reduction)\n",
			offKuniq.size(), k0.size()*fw.qOffset.size(), round(k0.size()*fw.qOffset.size()*1. / offKuniq.size()));
		logPrintf("Found %lu k-points with active states from %lu total k-points (%.0fx reduction)\n\n",
			nkSelected, nkTot, round(nkTot*1. / nkSelected));

		//Make group index, count, offset division and k division available on all processes of each group:
		if (mpiGroup->isHead()){
			iGroup = mpiGroupHead->iProcess();
			nGroups = mpiGroupHead->nProcesses();
			offStartGroup.assign(nGroups + 1, 0);
			ikStartGroup.assign(nGroups + 1, 0);
			TaskDivision groupDiv(offKuniq.size(), mpiGroupHead);
			for (size_t jGroup = 0; jGroup<nGroups; jGroup++){
				offStartGroup[jGroup + 1] = groupDiv.stop(jGroup);
				ikStartGroup[jGroup + 1] = ikStartOff[offStartGroup[jGroup + 1]];
			}
		}
		mpiGroup->bcast(iGroup);
		mpiGroup->bcast(nGroups);
		offStartGroup.resize(nGroups + 1);
		ikStartGroup.resize(nGroups + 1);
		mpiGroup->bcastData(offStartGroup);
		mpiGroup->bcastData(ikStartGroup);
	}

	//--------- Part 2: k-pair selection -------------
	std::vector<std::vector<size_t>> kpartners; //list of e-ph coupled k2 for each k1
	std::vector<double> kpairWeight; //Econserve weight factor for all k1 pairs due to downsampling (1 if no downsmapling)
	std::vector<std::pair<size_t, size_t>> kpairs; //pairs of k1 and k2
	size_t nActivePairs; //total number of active state pairs
	inline void selectActive(const double*& Ebegin, const double*& Eend, double Elo, double Ehi){ //narrow pointer range to data within [Estart,Estop]
		Ebegin = std::lower_bound(Ebegin, Eend, Elo);
		Eend = &(*std::lower_bound(reverse(Eend), reverse(Ebegin), Ehi, std::greater<double>())) + 1;
	}
	inline void kpSelect(const FeynWann::StatePh& state)
	{	//Find pairs of momentum conserving electron states with this q:
		for (size_t ik1 = 0; ik1 < k.size(); ik1++){
			const vector3<>& k1 = k[ik1];
			vector3<> k2 = k1 - state.q; //momentum conservation
			size_t ik2; if (not findK(k2, ik2)) continue;
			if (ik1 > ik2) continue;
			//Check energy conservation for pair of bands within active range:
			//--- determine ranges of all E1 and E2:
			const double *E1begin = E.data() + ik1*nBandsSel_probe, *E1end = E1begin + nBandsSel_probe;
			const double *E2begin = E.data() + ik2*nBandsSel_probe, *E2end = E2begin + nBandsSel_probe;
			//--- narrow to active energy ranges:
			selectActive(E1begin, E1end, Estart, Estop);
			selectActive(E2begin, E2end, Estart, Estop);
			//--- check energy ranges:
			bool Econserve = false;
			for (const double* E1 = E1begin; E1 < E1end; E1++){ //E1 in active range
				for (const double* E2 = E2begin; E2<E2end; E2++){ //E2 in active range
					for (const double omegaPh : state.omega) if (omegaPh>omegaPhCut){ //loop over non-zero phonon frequencies
						double deltaE_minus = (*E1) - (*E2) - omegaPh; //energy conservation violation
						double deltaE_plus = (*E1) - (*E2) + omegaPh; //energy conservation violation
						if (fabs(deltaE_minus) < nEphDelta*ePhDelta || fabs(deltaE_plus) < nEphDelta*ePhDelta){ //else negligible at the 10^-3 level for a Gaussian
							Econserve = true;
							nActivePairs++;
						}
					}
				}
			}
			if (Econserve) kpairs.push_back(std::make_pair(ik1, ik2));
		}
	}
	static void kpSelect(const FeynWann::StatePh& state, void* params){
		((LindbladInit*)params)->kpSelect(state);
	}
	void kpairSelect(const std::vector<vector3<>>& q0){
		logPrintf("\nFor kpairSelect: Estart= %lg Estop= %lg bStart= %d bStop= %d\n\n", Estart, Estop, bStart, bStop); logFlush();

		//Initialize kIndexMap for searching selected k-points:
		for (size_t ik = 0; ik < k.size(); ik++) kIndexMap[kIndex(k[ik])] = ik;

		//Parallel:
		size_t oStart, oStop; //!< range of offstes handled by this process groups
		if (mpiGroup->isHead()) TaskDivision(q0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart); mpiGroup->bcast(oStop);
		size_t noMine = oStop - oStart;
		size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

		//Find momentum-conserving k-pairs for which energy conservation is also possible for some bands:
		nActivePairs = 0;
		logPrintf("Scanning k-pairs with e-ph coupling: "); logFlush();
		for (size_t o = oStart; o<oStop; o++){
			fw.phLoop(q0[o], LindbladInit::kpSelect, this);
			if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		mpiWorld->allReduce(nActivePairs, MPIUtil::ReduceSum);

		//Synchronize selected kpairs across all processes:
		//--- determine nk on each process and compute cumulative counts
		std::vector<size_t> nkpPrev(mpiWorld->nProcesses() + 1);
		for (int jProc = 0; jProc < mpiWorld->nProcesses(); jProc++){
			size_t nkpCur = kpairs.size();
			mpiWorld->bcast(nkpCur, jProc); //nkCur = k.size() on jProc in all processes
			nkpPrev[jProc + 1] = nkpPrev[jProc] + nkpCur; //cumulative count
		}
		size_t nkpairs = nkpPrev.back();
		//--- broadcast kpairs:
		{	//Set kpairs in position in global arrays:
			std::vector<std::pair<size_t, size_t>> kpairs(nkpairs);
			std::copy(this->kpairs.begin(), this->kpairs.end(), kpairs.begin() + nkpPrev[mpiWorld->iProcess()]);
			//Broadcast:
			for (int jProc = 0; jProc < mpiWorld->nProcesses(); jProc++){
				size_t ikpStart = nkpPrev[jProc], nkp = nkpPrev[jProc + 1] - ikpStart;
				mpiWorld->bcast(((size_t*)kpairs.data()) + ikpStart * 2, nkp * 2, jProc);
			}
			//Store to class variables:
			std::swap(kpairs, this->kpairs);
		}
		//--- report:
		size_t nkpairsTot = k.size()*k.size();
		logPrintf("Found %lu k-pairs (ik <= jk) with e-ph coupling from %lu total pairs of selected k-points (%.0fx reduction)\n",
			nkpairs, nkpairsTot/2, round(nkpairsTot/2. / nkpairs));
		//--- initialize kpartners (list of k2 by k1):
		// jxu: I did not used the following part but I kept it
		kpartners.resize(k.size());
		for (auto kpair : kpairs)
			kpartners[kpair.first].push_back(kpair.second);
		size_t nPartnersMin = k.size(), nPartnersMax = 0;
		for (std::vector<size_t>& kp : kpartners){
			std::sort(kp.begin(), kp.end()); //sort k2 within each k1 array
			const size_t& nPartners = kp.size();
			if (nPartners < nPartnersMin) nPartnersMin = nPartners;
			if (nPartners > nPartnersMax) nPartnersMax = nPartners;
		}
		logPrintf("Selected %lu active state pairs (ik <= jk) from %lu total electronic state pairs (%.0fx reduction)\n",
			nActivePairs, (modeStop-modeStart)*nkpairs*(size_t)std::pow(nBandsSel, 2), round((modeStop-modeStart)*nkpairs*(size_t)std::pow(nBandsSel, 2)*1. / nActivePairs));
	}

	//--------- Part 3: Save data -------------
	string dir_ldbd = "ldbd_data/";
	string dir_debug = "debug_info/";
	void saveR(){
		if (mpiWorld->isHead()){
			int dim = 3;
			for (int idir = 0; idir < 3; idir++)
				if (fw.isTruncated[idir]) dim -= 1;
			string fname = dir_ldbd + "ldbd_R.dat";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "%d\n", dim);
			fw.R.print(fp, " %14.7le", false);
			fclose(fp);
		}
	}
	void saveSize(){
		if (mpiWorld->isHead()){
			string fname = dir_ldbd + "ldbd_size.dat";
			FILE *fp = fopen(fname.c_str(), "w");
			if (ePhOnlyElec)
				fprintf(fp, "There are scatterings only for conduction electrons\n");
			else if (Estop < Emid)
				fprintf(fp, "There are scatterings only for valence electrons\n");
			else
				fprintf(fp, "There are scatterings for both valence and conduction electrons\n");
			if (ePhOnlyElec)
				fprintf(fp, "%d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph_elec bTop_eph_elec\n", 
					bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, bBot_eph - bRef, bTop_eph - bRef);
			else if (ePhOnlyHole)
				fprintf(fp, "%d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph_elec bTop_eph_elec bBot_eph_hole bTop_eph_hole\n",
					bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, 0, 0, bBot_eph - bRef, bTop_eph - bRef);
			else
				fprintf(fp, "%d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph bTop_eph\n",
					bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, bBot_eph - bRef, bTop_eph - bRef);
			fprintf(fp, "%lu %lu # nk_full nk\n", nkTot, k.size());
			if (ePhOnlyElec)
				fprintf(fp, "%lu # nkpair_elec\n", kpairs.size());
			else if (ePhOnlyHole)
				fprintf(fp, "%d %lu # nkpair_elec nkpair_hole\n", 0, kpairs.size());
			else
				fprintf(fp, "%lu # nkpair\n", kpairs.size());
			fprintf(fp, "%d %d # modeStart modeStp\n", modeStart, modeStop);
			fprintf(fp, "%21.14le # T\n", Tmax); // Currently T = Tmax
			fprintf(fp, "%21.14le %21.14le # muMin, muMax\n", dmuMin, dmuMax);
			fprintf(fp, "%lg # degauss\n", ePhDelta / sqrt(2));
			fclose(fp);
		}
	}
	void saveEk(){
		//Maybe formatted file is better
		if (mpiWorld->isHead()){
			string fname = dir_ldbd + "ldbd_kvec.bin";
			FILE *fp = fopen(fname.c_str(), "wb");
			fwrite(k.data(), sizeof(double), k.size() * 3, fp);
			fclose(fp);
			fname = dir_ldbd + "ldbd_ek.bin";
			fp = fopen(fname.c_str(), "wb");
			fwrite(E.data(), sizeof(double), k.size() * nBandsSel_probe, fp);
			fclose(fp);
		}
	}
	std::vector<FeynWann::StateE> state_elec;
	void saveElec(){
		TaskDivision tasks(k.size(), mpiWorld);
		size_t ikStart, ikStop;
		tasks.myRange(ikStart, ikStop);
		FeynWann::StateE e;
		fw.eCalc(k[ikStart], e);
		state_elec.resize(k.size(), e);
		int kInterval = std::max(1, int(round(k.size() / 50.)));
		logPrintf("Compute StateE in parallel:"); logFlush();
		for (size_t ik = ikStart; ik < ikStop; ik++){
			fw.eCalc(k[ik], state_elec[ik]);
			if ((ik + 1) % kInterval == 0) { logPrintf("%d%% ", int(round((ik + 1)*100. / k.size()))); logFlush(); }
		}
		logPrintf(" done\n"); logFlush();

		for (int whose = 0; whose < mpiWorld->nProcesses(); whose++)
			for (int ik = tasks.start(whose); ik < tasks.stop(whose); ik++)
				fw.bcastState_inEphLoop(state_elec[ik], mpiWorld, whose);
		logPrintf("bcastState_inEphLoop done\n");

		if (mpiWorld->isHead()){
			//  write spin matrices
			string fnames = dir_ldbd + "ldbd_smat.bin";
			FILE *fps = fopen(fnames.c_str(), "wb");
			for (size_t ik = 0; ik < k.size(); ik++)
			for (size_t iDir = 0; iDir < 3; iDir++){
				matrix s = 0.5*state_elec[ik].S[iDir](bBot_dm, bTop_dm, bBot_dm, bTop_dm); // notice the band range here
				matrix st = transpose(s); // from ColMajor to RowMajor
				fwrite(st.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*(bTop_dm - bBot_dm), fps);
			}
			fclose(fps);
		}
		if (mpiWorld->isHead()){
			string fnamev = dir_ldbd + "ldbd_vmat.bin";
			FILE *fpv = fopen(fnamev.c_str(), "wb");
			for (size_t ik = 0; ik < k.size(); ik++)
			for (size_t iDir = 0; iDir < 3; iDir++){
				matrix v = state_elec[ik].v[iDir](bBot_dm, bTop_dm, bBot_probe, bTop_probe); // notice the band range here
				matrix vt = transpose(v); // from ColMajor to RowMajor
				fwrite(vt.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*nBandsSel_probe, fpv);
			}
			fclose(fpv);
		}
	}
	void saveKpair(){
		if (mpiWorld->isHead()){
			string fnamek = dir_ldbd + "ldbd_kpair_k1st.bin"; string fnamekp = dir_ldbd + "ldbd_kpair_k2nd.bin";
			FILE *fpk = fopen(fnamek.c_str(), "wb"), *fpkp = fopen(fnamekp.c_str(), "wb");
			for (size_t ikpair = 0; ikpair < kpairs.size(); ikpair++){
				fwrite(&kpairs[ikpair].first, sizeof(size_t), 1, fpk);
				fwrite(&kpairs[ikpair].second, sizeof(size_t), 1, fpkp);
			}
			fclose(fpk); fclose(fpkp);
		}
	}
	
	//--------- Part 4: Electron-Phonon related matrices -------------
	int ikpairStart, ikpairStop, nkpairMine;
	complex *P1, *P2, *App, *Apm, *Amp, *Amm, *A2pp, *A2pm, *A2mp, *A2mm;
	double **imsig;
	
	void compute_eph(){
		ostringstream convert; convert << mpiWorld->iProcess();
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname1 and fname2 are not created for non-root processes
		string fname1 = dir_ldbd + "ldbd_P1_lindblad.bin." + convert.str(), fname1c = dir_ldbd + "ldbd_P1_conventional.bin." + convert.str();
		string fname2 = dir_ldbd + "ldbd_P2_lindblad.bin." + convert.str(), fname2c = dir_ldbd + "ldbd_P2_conventional.bin." + convert.str();
		FILE *fp1 = fopen(fname1.c_str(), "wb"), *fp1c = fopen(fname1c.c_str(), "wb");
		FILE *fp2 = fopen(fname2.c_str(), "wb"), *fp2c = fopen(fname2c.c_str(), "wb");
		string fnamegm = dir_ldbd + "ldbd_gm.bin." + convert.str();
		FILE *fpgm; if (printgm) fpgm = fopen(fnamegm.c_str(), "wb");
		string fnamewq = dir_ldbd + "ldbd_wq.bin." + convert.str(); // phonon frequency of each pair (k,k')
		FILE *fpwq; if (printgm) fpwq = fopen(fnamewq.c_str(), "wb");
		string fnamesig = dir_ldbd + "ldbd_imsig.bin";
		FILE *fpsig = fopen(fnamesig.c_str(), "wb");
		bool ldebug = DEBUG;
		string fnamed = dir_debug + "ldbd_debug_compute_eph.out." + convert.str();
		FILE *fpd; if (ldebug) fpd = fopen(fnamed.c_str(), "w");

		// the index order is consistent with the file name order
		TaskDivision(kpairs.size(), mpiWorld).myRange(ikpairStart, ikpairStop);
		nkpairMine = ikpairStop - ikpairStart;
		size_t nkpairInterval = std::max(1, int(round(nkpairMine / 50.))); //interval for reporting progress
		MPI_Barrier(MPI_COMM_WORLD);

		App = alloc_array(nBandsSel*nBandsSel); Apm = alloc_array(nBandsSel*nBandsSel); Amp = alloc_array(nBandsSel*nBandsSel); Amm = alloc_array(nBandsSel*nBandsSel);
		A2pp = alloc_array(nBandsSel*nBandsSel); A2pm = alloc_array(nBandsSel*nBandsSel); A2mp = alloc_array(nBandsSel*nBandsSel); A2mm = alloc_array(nBandsSel*nBandsSel);
		P1 = alloc_array((int)std::pow(nBandsSel, 4)); P2 = alloc_array((int)std::pow(nBandsSel, 4));
		imsig = alloc_real_array(k.size(), nBandsSel);

		double nEphDelta_forJXU = nEphDelta * sqrt(2);
		double ePhDelta_forJXU = ePhDelta / sqrt(2); // just to stress ePhDelta is defined differently in different codes

		logPrintf("Compute EPH: \n"); logFlush();
		for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
			int ikpair_glob = ikpair_local + ikpairStart;
			int ik = kpairs[ikpair_glob].first, jk = kpairs[ikpair_glob].second;
			diagMatrix Ek = state_elec[ik].E(bStart, bStop), Ekp = state_elec[jk].E(bStart, bStop);
			FeynWann::StatePh php, phm;
			fw.phCalc(k[jk] - k[ik], php); // q = k' -k
			fw.phCalc(k[ik] - k[jk], phm); // q = k - k'
			FeynWann::MatrixEph mp, mm;
			fw.ePhCalc(state_elec[jk], state_elec[ik], php, mp); // g^-_k'k
			fw.ePhCalc(state_elec[ik], state_elec[jk], phm, mm); // g^-_kk'

			if (printgm){
				for (int alpha = modeStart; alpha < modeStop; alpha++){
					matrix gm = mm.M[alpha](bStart, bStop, bStart, bStop);
					fwrite(gm.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpgm);
				}
				fwrite(phm.omega.data(), sizeof(double), (modeStop - modeStart), fpwq);
				if (ik < jk){
					for (int alpha = modeStart; alpha < modeStop; alpha++){
						matrix gp = dagger(mp.M[alpha](bStart, bStop, bStart, bStop));
						fwrite(gp.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpgm);
					}
					fwrite(php.omega.data(), sizeof(double), (modeStop - modeStart), fpwq);
				}
			}


			if (ldebug)
				fprintf(fpd, "\nik= %d, k= %lg %lg %lg, ikp= %d, kp= %lg %lg %lg\n",
					ik, k[ik][0], k[ik][1], k[ik][2], jk, k[jk][0], k[jk][1], k[jk][2]);

			compute_P(ik, jk, Ek, Ekp, php, phm, mp, mm,
				ePhDelta_forJXU, nEphDelta_forJXU, true, fp1, fp2, ldebug, fpd, true, false); // gaussian smearing
			compute_P(ik, jk, Ek, Ekp, php, phm, mp, mm,
				ePhDelta_forJXU, nEphDelta_forJXU, false, fp1c, fp2c, ldebug, fpd, false, false); // conventional, gaussian smearing

			//Print progress:
			if ((ikpair_local + 1) % nkpairInterval == 0) { logPrintf("%d%% ", int(round((ikpair_local + 1)*100. / nkpairMine))); logFlush(); }
		}
		for (size_t ik = 0; ik < k.size(); ik++)
			mpiWorld->allReduce(&imsig[ik][0], nBandsSel, MPIUtil::ReduceSum);
		if (mpiWorld->isHead()){
			for (size_t ik = 0; ik < k.size(); ik++)
				fwrite(imsig[ik], sizeof(double), nBandsSel, fpsig);
			write_imsige();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		logPrintf("done.\n"); logFlush();
		fclose(fp1); fclose(fp2); fclose(fp1c); fclose(fp2c); if (printgm) { fclose(fpgm); fclose(fpwq); } fclose(fpsig); if (ldebug) fclose(fpd);
	}

	void compute_P(int ik, int jk, diagMatrix& Ek, diagMatrix& Ekp, FeynWann::StatePh& php, FeynWann::StatePh& phm, FeynWann::MatrixEph& mp, FeynWann::MatrixEph& mm,
		double degauss, double ndegauss, bool compute_imsig, FILE* fp1, FILE* fp2, bool ldebug, FILE* fpd, bool lindblad, bool lorentzian){
		ldebug = ldebug && lindblad && !lorentzian;
		// compute_imshig should only be true for one of compute_P in subroutine compute_eph
		double ethr = degauss * ndegauss;
		zeros(P1, (int)std::pow(nBandsSel, 4)); zeros(P2, (int)std::pow(nBandsSel, 4));

		for (int alpha = modeStart; alpha < modeStop; alpha++){
			double wq = phm.omega[alpha];
			double sigma = !variedDelta ? degauss : std::min(degauss, wq / (ndegauss / sqrt(2) + 1) / sqrt(2)); // to be consistent with Shankar's code
			double sigma2 = std::pow(sigma, 2);
			double prefac_sqrtexp, prefac_sqrtdelta, prefac_exp, prefac_delta, deltaplus, deltaminus;
			prefac_sqrtexp = -0.25 / sigma2; prefac_exp = -0.5 / sigma2;
			if (lorentzian)
				prefac_delta = sigma / M_PI;
			else
				prefac_delta = 1. / (sigma * sqrt(2.*M_PI));
			prefac_sqrtdelta = sqrt(prefac_delta);
			double prefac_imsig = M_PI / nkTot * prefac_delta;

			double nq0 = bose(std::max(1e-3, wq / Tmax));
			//if (ldebug) fprintf(fpd, "alpha= %d wq= %lg nq= %lg\n", alpha, wqm, nqm);
			matrix gp = dagger(mp.M[alpha](bStart, bStop, bStart, bStop)); // g^+_kk' = g^-_k'k^(dagger_n)
			matrix gm = mm.M[alpha](bStart, bStop, bStart, bStop);
			/*
			if (ldebug){
			//fprintf(fpd, "gp:");
			//for (int b1 = 0; b1 < nBandsSel; b1++)
			//for (int b2 = 0; b2 < nBandsSel; b2++)
			//	fprintf(fpd, " (%lg,%lg)", gp(b1, b2).real(), gp(b1, b2).imag());
			//fprintf(fpd, "\n:");
			fprintf(fpd, "gm:");
			for (int b1 = 0; b1 < nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel; b2++)
			fprintf(fpd, " (%lg,%lg)", gm(b1, b2).real(), gm(b1, b2).imag());
			fprintf(fpd, "\n");
			}
			*/
			bool conserve = false;
			//G^+- = g^+- sqrt(delta(ek - ekp +- wq)); G^+_kk' = G^-_k'k^(dagger_n)
			for (int b1 = 0; b1 < nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel; b2++){
				complex Gp = c0, G2p = c0, Gm = c0, G2m = c0;
				double dE = Ek[b1] - Ekp[b2];
				// phonon emission
				double nq = !detailBalance ? nq0 : bose(std::max(1e-3, -dE / Tmax));
				if (fabs(dE + wq) < ethr && (antiResonant || dE < 0.)){
					conserve = true;
					if (lorentzian)
						deltaplus = 1. / (std::pow(dE + wq, 2) + sigma2);
					else
						deltaplus = exp(prefac_exp*std::pow(dE + wq, 2));
					if (lindblad)
						G2p = prefac_sqrtdelta * gp(b1, b2) * sqrt(deltaplus);
					else
						G2p = prefac_delta * gp(b1, b2) * deltaplus;

					if (compute_imsig){
						imsig[ik][b1] += prefac_imsig * gp(b1, b2).norm() * (F[jk][b2] + nq) * deltaplus;
						if (ik < jk) imsig[jk][b2] += prefac_imsig * gp(b1, b2).norm() * (1 - F[ik][b1] + nq) * deltaplus;
					}
				}
				if (lindblad)
					Gp = G2p;
				else
					Gp = gp(b1, b2);
				App[b1*nBandsSel + b2] = Gp * sqrt(nq + 1);
				Apm[b1*nBandsSel + b2] = Gp * sqrt(nq);
				A2pp[b1*nBandsSel + b2] = G2p * sqrt(nq + 1);
				A2pm[b1*nBandsSel + b2] = G2p * sqrt(nq);

				// phonon absorption
				nq = !detailBalance ? nq0 : bose(std::max(1e-3, dE / Tmax));
				if (fabs(dE - wq) < ethr && (antiResonant || dE > 0.)){
					conserve = true;
					double deltaminus;
					if (lorentzian)
						deltaminus = 1. / (std::pow(dE - wq, 2) + sigma2);
					else
						deltaminus = exp(prefac_exp*std::pow(dE - wq, 2));
					if (lindblad)
						G2m = prefac_sqrtdelta * gm(b1, b2) * sqrt(deltaminus);
					else
						G2m = prefac_delta * gm(b1, b2) * deltaminus;

					if (compute_imsig){
						imsig[ik][b1] += prefac_imsig * gm(b1, b2).norm() * (1 - F[jk][b2] + nq) * deltaminus;
						if (ik < jk) imsig[jk][b2] += prefac_imsig * gm(b1, b2).norm() * (F[ik][b1] + nq) * deltaminus;
					}
				}
				if (lindblad)
					Gm = G2m;
				else
					Gm = gm(b1, b2);
				Amp[b1*nBandsSel + b2] = Gm * sqrt(nq + 1);
				Amm[b1*nBandsSel + b2] = Gm * sqrt(nq);
				A2mp[b1*nBandsSel + b2] = G2m * sqrt(nq + 1);
				A2mm[b1*nBandsSel + b2] = G2m * sqrt(nq);
			} // loop on b1 and b2

			if (conserve){
				// P1_n3n2,n4n5 = G^+-_n3n4 * conj(G^+-_n2n5) * nq^+-
				// P2_n3n4,n1n5 = G^-+_n1n3 * conj(G^-+_n5n4) * nq^+-
				for (int i1 = 0; i1 < nBandsSel; i1++)
				for (int i2 = 0; i2 < nBandsSel; i2++){
					int n12 = (i1*nBandsSel + i2)*nBandsSel*nBandsSel;
					for (int i3 = 0; i3 < nBandsSel; i3++){
						int i13 = i1*nBandsSel + i3;
						int i31 = i3*nBandsSel + i1;
						for (int i4 = 0; i4 < nBandsSel; i4++){
							// notice that ik <= jk, (jk,ik) can be obtained from
							// conj(P2_k'k_n1n2n3n4) = P1_kk'_n1n2n3n4 and conj(P1_k'k_n1n2n3n4) = P2_kk'_n1n2n3n4 (even for conventional)
							P1[n12 + i3*nBandsSel + i4] += App[i13] * conj(A2pp[i2*nBandsSel + i4]) + Amm[i13] * conj(A2mm[i2*nBandsSel + i4]);
							P2[n12 + i3*nBandsSel + i4] += Amp[i31] * conj(A2mp[i4*nBandsSel + i2]) + Apm[i31] * conj(A2pm[i4*nBandsSel + i2]);
						}
					}
				}
			} // if (conserve)
		} // loop on alpha

		if (ldebug){
			///*
			fprintf(fpd, "P1:\n");
			for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++)
			fprintf(fpd, " (%lg,%lg)", P1[b1*nBandsSel*nBandsSel + b2].real(), P1[b1*nBandsSel*nBandsSel + b2].imag());
			fprintf(fpd, "\nP2:\n");
			for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++)
			fprintf(fpd, " (%lg,%lg)", P2[b1*nBandsSel*nBandsSel + b2].real(), P2[b1*nBandsSel*nBandsSel + b2].imag());
			fprintf(fpd, "\n");
			//*/
		}

		fwrite(P1, 2 * sizeof(double), (int)std::pow(nBandsSel, 4), fp1);
		fwrite(P2, 2 * sizeof(double), (int)std::pow(nBandsSel, 4), fp2);
	}
	void write_imsige(){
		string fnamesigkn = dir_ldbd + "ldbd_imsigkn.out";
		FILE *fpsigkn = fopen(fnamesigkn.c_str(), "w");
		fprintf(fpsigkn, "E(eV) ImSigma_kn(eV)\n");
		double imsig_max = imsig[0][0], imsig_min = imsig[0][0];
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			for (int b = 0; b < nBandsSel; b++){
				fprintf(fpsigkn, "%14.7le %14.7le\n", Ek[b] / eV, imsig[ik][b] / eV);
				if (imsig[ik][b] > imsig_max) imsig_max = imsig[ik][b];
				if (imsig[ik][b] < imsig_min) imsig_min = imsig[ik][b];
			}
		}
		logPrintf("imsig_min = %lg eV imsig_max = %lg eV\n", imsig_min / eV, imsig_max / eV); logFlush();
		fclose(fpsigkn);

		std::vector<double> imsige(102); std::vector<int> nstate(102);
		double dE = (Estop - Estart) / 100;
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			for (int b = 0; b < nBandsSel; b++){
				int ie = round((Ek[b]-Estart) / dE);
				if (ie >= 0 && ie <= 101){
					nstate[ie]++;
					imsige[ie] += imsig[ik][b];
				}
			}
		}
		string fnamesige = dir_ldbd + "ldbd_imsige.out";
		FILE *fpsige = fopen(fnamesige.c_str(), "w");
		fprintf(fpsige, "E(eV) ImSigma(eV) N_States\n");
		for (int ie = 0; ie < 102; ie++){
			if (nstate[ie] > 0){
				imsige[ie] /= nstate[ie];
				fprintf(fpsige, "%14.7le %14.7le %d\n", (Estart + ie*dE) / eV, imsige[ie] / eV, nstate[ie]);
			}
		}
		fclose(fpsige);
	}
	void merge_eph_P(){
		// This subroutine requires that the index order is consistent with the file name order
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpiWorld->isHead()){
			logPrintf("Merge P1 and P2:\nprocessing file ");
			string fname1 = dir_ldbd + "ldbd_P1_lindblad.bin", fname1c = dir_ldbd + "ldbd_P1_conventional.bin";
			string fname2 = dir_ldbd + "ldbd_P2_lindblad.bin", fname2c = dir_ldbd + "ldbd_P2_conventional.bin";

			std::vector<complex> work((size_t)std::pow(nBandsSel, 4));
			for (int i = 0; i < mpiWorld->nProcesses(); i++){
				ostringstream convert; convert << i;
				string fname1i = dir_ldbd + "ldbd_P1_lindblad.bin." + convert.str(), fname1ic = dir_ldbd + "ldbd_P1_conventional.bin." + convert.str();
				string fname2i = dir_ldbd + "ldbd_P2_lindblad.bin." + convert.str(), fname2ic = dir_ldbd + "ldbd_P2_conventional.bin." + convert.str();

				int nkpairMine1 = read_append_file(fname1i, fname1, work, (size_t)std::pow(nBandsSel, 4));
				int nkpairMine2 = read_append_file(fname2i, fname2, work, (size_t)std::pow(nBandsSel, 4));
				read_append_file(fname1ic, fname1c, work, (size_t)std::pow(nBandsSel, 4));
				read_append_file(fname2ic, fname2c, work, (size_t)std::pow(nBandsSel, 4));

				if (i%10 == 0) printf("%d ", i);
			}
			logPrintf("done\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	size_t read_append_file(string fnamein, string fnameout, std::vector<complex>& v, size_t n){
		FILE *fpin = fopen(fnamein.c_str(), "rb");
		FILE *fpout = fopen(fnameout.c_str(), "ab");
		size_t nline = 0;
		while (fread(v.data(), 2 * sizeof(double), n, fpin) == n){
			nline++;
			fwrite(v.data(), 2 * sizeof(double), n, fpout);
		}
		fclose(fpin); fclose(fpout);
		remove(fnamein.c_str());
		return nline;
	}
	void merge_eph_gm(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (printgm && mpiWorld->isHead()){
			logPrintf("Merge gm (currently delete):\n");
			string fname1 = dir_ldbd + "ldbd_P1_lindblad.bin", fname1c = dir_ldbd + "ldbd_P1_conventional.bin";
			string fname2 = dir_ldbd + "ldbd_P2_lindblad.bin", fname2c = dir_ldbd + "ldbd_P2_conventional.bin";
			for (int i = 0; i < mpiWorld->nProcesses(); i++){
				ostringstream convert; convert << i;
				string fnamegmi = dir_ldbd + "ldbd_gm.bin." + convert.str();
				remove(fnamegmi.c_str());
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	std::vector<diagMatrix> F;
	void computeF(){
		double carrier_density = 0., au2cm = 5.291772109038e-9;
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop), Fk(nBandsSel);
			for (int b = 0; b < nBandsSel; b++){
				Fk[b] = fermi((Ek[b] - dmuMax) / Tmax);
				if (dmuMax > Emid && Ek[b] > Emid) carrier_density += Fk[b];
				if (dmuMax < Emid && Ek[b] < Emid) carrier_density += 1. - Fk[b];
			}
			F.push_back(Fk);
		}
		carrier_density /= nkTot;
		double dimension = 3;
		for (int iDir = 0; iDir<3; iDir++)
			if (fw.isTruncated[iDir]) dimension--;
		double volume = fabs(det(fw.R));
		if (dimension == 3){
			carrier_density /= volume * std::pow(au2cm, 3);
			logPrintf("Carrier density: %14.7le cm^-3\n", carrier_density);
		}
		else if (dimension == 2){
			double area = volume / fw.R(2, 2);
			carrier_density /= area * std::pow(au2cm, 2);
			logPrintf("Carrier density: %14.7le cm^-2\n", carrier_density);
		}
		else if (dimension == 1){
			double height = fw.R(2, 2);
			carrier_density /= height * au2cm;
			logPrintf("Carrier density: %14.7le cm^-1\n", carrier_density);
		}
		else
			logPrintf("Carrier density: %14.7le\n", carrier_density);
	}
	void saveData(){
		// please never use -G? when running this program
		if (mpiWorld->isHead()) system("mkdir ldbd_data");
		if (mpiWorld->isHead()) system("mkdir debug_info");

		saveR(); logPrintf("saveR done\n");
		saveSize(); logPrintf("saveSize done\n");
		saveEk(); logPrintf("saveEk done\n");
		saveElec(); logPrintf("saveElec done\n");
		saveKpair(); logPrintf("saveKpair done\n");

		logPrintf("\nFor ePh: Estart= %lg eV Estop= %lg eV bStart= %d bStop= %d\n\n", Estart/eV, Estop/eV, bStart, bStop); logFlush();
		computeF();	// prepare fermi-dirac occupations and compute carrier concentration

		compute_eph(); // save scattering matrices
	}

	//--------- Part 5: Spin relaxation -------------
	complex **dm, **dm1, **ddm, *ddmdt_contrib, *maux1, *maux2;
	complex *P1_next, *P2_next;
	double prefac_eph;
	
	inline void term1_P(complex *dm1, complex *p, complex *dm){
		// + (1-dm_k)_n1n3 * P1_kk'_n3n2,n4n5 * dm_k'_n4n5
		// maux1 = P1_kk'_n3n2,n4n5 * dm_k'_n4n5
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nBandsSel*nBandsSel, 1, nBandsSel*nBandsSel, &c1, p, nBandsSel*nBandsSel, dm, 1, &c0, maux1, 1);
		// maux2 = (1-dm_k)_n1n3 * maux1
		cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, nBandsSel, nBandsSel, &c1, dm1, nBandsSel, maux1, nBandsSel, &c0, maux2, nBandsSel);
		for (int i = 0; i < nBandsSel*nBandsSel; i++)
			ddmdt_contrib[i] += maux2[i];
	}
	inline void term2_P(complex *dm1, complex *p, complex *dm){
		// - (1-dm_k')_n3n4 * P2_kk'_n3n4,n1n5 * dm_k_n5n2
		// maux1 = (1-dm_k')_n3n4 * P2_kk'_n3n4,n1n5
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, nBandsSel*nBandsSel, nBandsSel*nBandsSel, &c1, dm1, nBandsSel*nBandsSel, p, nBandsSel*nBandsSel, &c0, maux1, nBandsSel*nBandsSel);
		// maux2 = maux1 * dm_k_n5n2
		cblas_zhemm(CblasRowMajor, CblasRight, CblasUpper, nBandsSel, nBandsSel, &c1, dm, nBandsSel, maux1, nBandsSel, &c0, maux2, nBandsSel);
		for (int i = 0; i < nBandsSel*nBandsSel; i++)
			ddmdt_contrib[i] -= maux2[i];
	}
	void T1_1step_useP(bool lindblad, double Bzpert){
		ostringstream convert; convert << mpiWorld->iProcess();
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname1 and fname2 are not created for non-root processes
		string fname1, fname2;
		if (lindblad){
			fname1 = dir_ldbd + "ldbd_P1_lindblad.bin." + convert.str();
			fname2 = dir_ldbd + "ldbd_P2_lindblad.bin." + convert.str();
		}
		else{
			fname1 = dir_ldbd + "ldbd_P1_conventional.bin." + convert.str();
			fname2 = dir_ldbd + "ldbd_P2_conventional.bin." + convert.str();
		}
		FILE *fp1 = fopen(fname1.c_str(), "rb");
		FILE *fp2 = fopen(fname2.c_str(), "rb");
		string fnamed = "ldbd_debug_T1_1step_useP.out." + convert.str();
		bool ldebug = false && lindblad && fabs(Bzpert) < 1e-10 * Tesla;
		FILE *fpd; if (ldebug) fpd = fopen(fnamed.c_str(), "w");

		init_dm(dm, k.size(), nBandsSel, F);
		double szeq = compute_sz(dm, k.size(), nkTot, nBandsSel, bStart, bStop, state_elec);
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			matrix H1 = Bzpert * state_elec[ik].S[2](bStart, bStop, bStart, bStop);
			matrix dRho = dRho_H1(Ek, F[ik], Tmax, H1, nBandsSel);
			for (int i = 0; i < nBandsSel; i++)
			for (int j = 0; j < nBandsSel; j++)
				dm[ik][i*nBandsSel + j] += dRho(i, j);
		}
		double sz0 = compute_sz(dm, k.size(), nkTot, nBandsSel, bStart, bStop, state_elec);
		set_dm1(dm, k.size(), nBandsSel, dm1);
		zeros(ddm, k.size(), nBandsSel*nBandsSel);

		P1_next = alloc_array((int)std::pow(nBandsSel, 4)); P2_next = alloc_array((int)std::pow(nBandsSel, 4));

		prefac_eph = 2 * M_PI / nkTot;
		for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
			int ikpair_glob = ikpair_local + ikpairStart;
			int ik_glob = kpairs[ikpair_glob].first, ikp_glob = kpairs[ikpair_glob].second;

			if (ldebug)
				fprintf(fpd, "\nk= %lg %lg %lg, kp= %lg %lg %lg\n",
					k[ik_glob][0], k[ik_glob][1], k[ik_glob][2], k[ikp_glob][0], k[ikp_glob][1], k[ikp_glob][2]);

			if (fread(P1, 2 * sizeof(double), std::pow(nBandsSel, 4), fp1) != std::pow(nBandsSel, 4))
				error_message("error during reading P1", "T1_1step_useP");
			if (fread(P2, 2 * sizeof(double), std::pow(nBandsSel, 4), fp2) != std::pow(nBandsSel, 4))
				error_message("error during reading P2", "T1_1step_useP");

			compute_ddm(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], P1, P2, ddm[ik_glob]);

			///*
			if (ldebug){
				fprintf(fpd, "P1:\n");
				for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
				for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++)
					fprintf(fpd, " (%lg,%lg)", P1[b1*nBandsSel*nBandsSel + b2].real(), P1[b1*nBandsSel*nBandsSel + b2].imag());
				fprintf(fpd, "\nP2:\n");
				for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
				for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++)
					fprintf(fpd, " (%lg,%lg)", P2[b1*nBandsSel*nBandsSel + b2].real(), P2[b1*nBandsSel*nBandsSel + b2].imag());
				fprintf(fpd, "\n");
				fprintf(fpd, "\nddmdt_contrib:\n");
				for (int b1 = 0; b1 < nBandsSel; b1++)
				for (int b2 = 0; b2 < nBandsSel; b2++)
					fprintf(fpd, " (%lg,%lg)", ddmdt_contrib[b1*nBandsSel + b2].real(), ddmdt_contrib[b1*nBandsSel + b2].imag());
				fprintf(fpd, "\n");
				fprintf(fpd, "\nddm[ik_glob]:\n");
				for (int b1 = 0; b1 < nBandsSel; b1++)
				for (int b2 = 0; b2 < nBandsSel; b2++)
					fprintf(fpd, " (%lg,%lg)", ddm[ik_glob][b1*nBandsSel + b2].real(), ddm[ik_glob][b1*nBandsSel + b2].imag());
				fprintf(fpd, "\n");
			}
			//*/

			// compute (ikp, ik) contribution
			if (ik_glob < ikp_glob){
				for (int i = 0; i < (int)std::pow(nBandsSel, 4); i++){
					P1_next[i] = conj(P2[i]); P2_next[i] = conj(P1[i]);
				}
				compute_ddm(dm[ikp_glob], dm[ik_glob], dm1[ikp_glob], dm1[ik_glob], P1_next, P2_next, ddm[ikp_glob]);

				if (ldebug){
					fprintf(fpd, "P1_next:\n");
					for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
					for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++)
						fprintf(fpd, " (%lg,%lg)", P1_next[b1*nBandsSel*nBandsSel + b2].real(), P1_next[b1*nBandsSel*nBandsSel + b2].imag());
					fprintf(fpd, "\nP2_next:\n");
					for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
					for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++)
						fprintf(fpd, " (%lg,%lg)", P2_next[b1*nBandsSel*nBandsSel + b2].real(), P2_next[b1*nBandsSel*nBandsSel + b2].imag());
					fprintf(fpd, "\n");
					fprintf(fpd, "\nddmdt_contrib:\n");
					for (int b1 = 0; b1 < nBandsSel; b1++)
					for (int b2 = 0; b2 < nBandsSel; b2++)
						fprintf(fpd, " (%lg,%lg)", ddmdt_contrib[b1*nBandsSel + b2].real(), ddmdt_contrib[b1*nBandsSel + b2].imag());
					fprintf(fpd, "\n");
					fprintf(fpd, "\nddm[ikp_glob]:\n");
					for (int b1 = 0; b1 < nBandsSel; b1++)
					for (int b2 = 0; b2 < nBandsSel; b2++)
						fprintf(fpd, " (%lg,%lg)", ddm[ikp_glob][b1*nBandsSel + b2].real(), ddm[ikp_glob][b1*nBandsSel + b2].imag());
					fprintf(fpd, "\n");
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		for (size_t ik = 0; ik < k.size(); ik++){
			mpiWorld->allReduce(&ddm[ik][0], nBandsSel*nBandsSel, MPIUtil::ReduceSum);
		}
		for (size_t ik = 0; ik < k.size(); ik++)
		for (int b1 = 0; b1 < nBandsSel; b1++)
		for (int b2 = 0; b2 < nBandsSel; b2++)
			dm[ik][b1*nBandsSel + b2] += ddm[ik][b1*nBandsSel + b2];
		double sz1 = compute_sz(dm, k.size(), nkTot, nBandsSel, bStart, bStop, state_elec);
		const double ps = 1e3*fs; //picosecond
		logPrintf("dSz = %lg Szdot = %lg T1z = %lg ps\n", sz0 - szeq, sz1 - sz0, -(sz0 - szeq) / (sz1 - sz0) / ps); logFlush();
		fclose(fp1); fclose(fp2); if (ldebug) fclose(fpd);
	}
	void compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, complex *p1, complex *p2, complex *ddmk){
		zeros(ddmdt_contrib, nBandsSel*nBandsSel);
		term1_P(dm1k, p1, dmkp);
		term2_P(dm1kp, p2, dmk);
		for (int i = 0; i < nBandsSel; i++)
		for (int j = 0; j < nBandsSel; j++)
			ddmk[i*nBandsSel + j] += (prefac_eph*0.5) * (ddmdt_contrib[i*nBandsSel + j] + conj(ddmdt_contrib[j*nBandsSel + i]));
	}
	
	inline matrix dRho_H1(const diagMatrix& E, const diagMatrix& F, const double& T, const matrix& H1, const int& nBandsSel){
		matrix result(nBandsSel, nBandsSel); complex *rData = result.data();
		double invT = 1. / T;
		for (int b2 = 0; b2 < nBandsSel; b2++)
		for (int b1 = 0; b1 < nBandsSel; b1++){
			if (fabs(E[b1] - E[b2]) <= degthr){
				double Favg = 0.5 * (F[b1] + F[b2]);
				*rData = Favg * (Favg - 1.) * invT * H1(b1, b2);
			}
			else{
				*rData = (F[b1] - F[b2]) / (E[b1] - E[b2]) * H1(b1, b2);
			}
			rData++;
		}
		return result;
	}

	void degProj(matrix& M, diagMatrix& E, int nBandsSel, matrix& Mdeg){
		complex *MdegData = Mdeg.data();
		for (int b2 = 0; b2 < nBandsSel; b2++)
		for (int b1 = 0; b1 < nBandsSel; b1++){
			if (fabs(E[b1] - E[b2]) >= degthr){
				*MdegData = c0;
			}
			else {
				*MdegData = M(b1, b2);
			}
			MdegData++;
		}
	}
	void T1_rate_dRho(bool drhosdeg, bool obsdeg, bool imsigma_eph, double fac_imsig, double impurity_broadening){ // impurity_broadening in eV
		ostringstream convert; convert << mpiWorld->iProcess();
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname1 and fname2 are not created for non-root processes
		string fnamegm = dir_ldbd + "ldbd_gm.bin." + convert.str();
		FILE *fpgm = fopen(fnamegm.c_str(), "rb");

		std::vector<std::vector<matrix>> Sfull(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		std::vector<std::vector<matrix>> Sdeg(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		std::vector<std::vector<matrix>> dRho(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			for (int id = 0; id < 3; id++){
				Sfull[ik][id] = state_elec[ik].S[id](bStart, bStop, bStart, bStop);
				degProj(Sfull[ik][id], Ek, nBandsSel, Sdeg[ik][id]);
				if (drhosdeg)
					dRho[ik][id] = dRho_H1(Ek, F[ik], Tmax, Sdeg[ik][id], nBandsSel);
				else
					dRho[ik][id] = dRho_H1(Ek, F[ik], Tmax, Sfull[ik][id], nBandsSel);
			}
		}
		vector3<> dS = compute_spin(dRho, k.size(), nkTot, nBandsSel, bStart, bStop, state_elec);

		vector3<> Sdot(0., 0., 0.);
		double prefac = 2 * M_PI / nkTot / nkTot;
		double nEphDelta_forJXU = nEphDelta * sqrt(2);
		//if (ldebug) nEphDelta_forJXU = 4 * sqrt(2);
		double ePhDelta_forJXU = ePhDelta / sqrt(2); // just to stress ePhDelta is defined differently in different codes
		double ethr = nEphDelta_forJXU * ePhDelta_forJXU;

		for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
			int ikpair_glob = ikpair_local + ikpairStart;
			int ik0 = kpairs[ikpair_glob].first, jk0 = kpairs[ikpair_glob].second;

			// we need this trick since gm file stores g(jk,ik) right after g(ik,jk) if ik < jk but kpairs array stores only the pair (ik,jk) satisfying ik <= jk
			int nrun = (ik0 == jk0) ? 1 : 2;
			for (int irun = 0; irun < nrun; irun++){
				int ik, jk;
				if (irun == 0){	ik = ik0; jk = jk0; }
				else{ ik = jk0; jk = ik0; }

				diagMatrix Ek = state_elec[ik].E(bStart, bStop), Ekp = state_elec[jk].E(bStart, bStop);
				FeynWann::StatePh ph;
				fw.phCalc(k[ik] - k[jk], ph); // q = k - k'

				std::vector<double> prefac_sqrtexp(nBandsSel*nBandsSel), prefac_sqrtdelta(nBandsSel*nBandsSel);
				int bIndex = 0;
				for (int b2 = 0; b2 < nBandsSel; b2++)
				for (int b1 = 0; b1 < nBandsSel; b1++){
					double sigma = fac_imsig * (imsig[ik][b1] + imsig[jk][b2]) + impurity_broadening * eV;
					sigma = std::max(sigma, 1e-6*eV);
					prefac_sqrtexp[bIndex] = -0.25 / std::pow(sigma, 2);
					prefac_sqrtdelta[bIndex] = 1. / sqrt(sigma * sqrt(2.*M_PI));
					bIndex++;
				}

				vector3<> contrib(0., 0., 0.);
				for (int im = modeStart; im < modeStop; im++){
					double wq = ph.omega[im];
					double nq = bose(std::max(1e-3, wq / Tmax));
					if (!imsigma_eph){
						double sigma = !variedDelta ? ePhDelta_forJXU : std::min(ePhDelta_forJXU, wq / (nEphDelta_forJXU / sqrt(2) + 1) / sqrt(2)); // to be consistent with Shankar's code
						bIndex = 0;
						for (int b2 = 0; b2 < nBandsSel; b2++)
						for (int b1 = 0; b1 < nBandsSel; b1++){
							prefac_sqrtexp[bIndex] = -0.25 / std::pow(sigma, 2);
							prefac_sqrtdelta[bIndex] = 1. / sqrt(sigma * sqrt(2.*M_PI));
							bIndex++;
						}
					}

					matrix gm(nBandsSel, nBandsSel), gtmp(nBandsSel, nBandsSel);
					if (fread(gtmp.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpgm) == nBandsSel*nBandsSel){}
					else { error_message("error during reading gm", "T1_rate_dRho"); }

					if (irun == 0) gm = gtmp;
					else gm = dagger(gtmp);

					matrix G(nBandsSel, nBandsSel); complex *GData = G.data();
					bool conserve = false;
					bIndex = 0;
					for (int b2 = 0; b2 < nBandsSel; b2++)
					for (int b1 = 0; b1 < nBandsSel; b1++){
						double dE = Ek[b1] - Ekp[b2];
						if (fabs(dE - wq) < ethr && (antiResonant || dE > 0.)){
							conserve = true;
							double sqrtdeltaminus = exp(prefac_sqrtexp[bIndex] * std::pow(dE - wq, 2));
							*GData = prefac_sqrtdelta[bIndex] * gm(b1, b2) * sqrtdeltaminus;
						}
						else
							*GData = complex(0, 0);
						GData++; bIndex++;
					} // loop on b1 and b2

					if (!conserve) continue;
					std::vector<matrix> SGcomm(3, matrix(nBandsSel, nBandsSel));
					for (int id = 0; id < 3; id++)
					if (obsdeg)
						SGcomm[id] = Sdeg[ik][id] * G - G * Sdeg[jk][id];
					else
						SGcomm[id] = Sfull[ik][id] * G - G * Sfull[jk][id];

					diagMatrix nFi(nBandsSel), nFjbar(nBandsSel);
					for (int b = 0; b < nBandsSel; b++)
						nFi[b] = nq + F[ik][b];
					for (int b = 0; b < nBandsSel; b++)
						nFjbar[b] = nq + 1. - F[jk][b];
					std::vector<matrix> dRhoGcomm(3, matrix(nBandsSel, nBandsSel));
					for (int id = 0; id < 3; id++)
						dRhoGcomm[id] = dRho[ik][id] * G * nFjbar - nFi * G * dRho[jk][id];

					for (int id = 0; id < 3; id++)
					for (int b2 = 0; b2 < nBandsSel; b2++)
					for (int b1 = 0; b1 < nBandsSel; b1++)
						contrib[id] -= (SGcomm[id](b1, b2).conj() * dRhoGcomm[id](b1, b2)).real();
				}
				Sdot += prefac * contrib;
			}
		}

		mpiWorld->allReduce(Sdot, MPIUtil::ReduceSum);

		vector3<> T1;
		for (int id = 0; id < 3; id++){ T1[id] = -dS[id] / Sdot[id]; }
		const double ps = 1e3*fs; //picosecond
		logPrintf("\ndrhosdeg = %d obsdeg = %d imsigma_eph = %d fac_imsig = %lg impurity_broadening = %lg eV\n", drhosdeg, obsdeg, imsigma_eph, fac_imsig, impurity_broadening);
		logPrintf("dS[2] = %lg Sdot[2] = %lg T1 = %lg %lg %lg ps\n", dS[2], Sdot[2], T1[0] / ps, T1[1] / ps, T1[2] / ps); logFlush();
		fclose(fpgm);
	}

	void relax_1step_useP(){
		// allocations for real-time dynamics
		dm = alloc_array(k.size(), nBandsSel*nBandsSel);
		dm1 = alloc_array(k.size(), nBandsSel*nBandsSel);
		ddm = alloc_array(k.size(), nBandsSel*nBandsSel);
		ddmdt_contrib = alloc_array(nBandsSel*nBandsSel);
		maux1 = alloc_array(nBandsSel*nBandsSel);
		maux2 = alloc_array(nBandsSel*nBandsSel);

		T1_1step_useP(true, 0); // just to confirm zero dRho leads to zero change
		T1_1step_useP(false, 0); // just to confirm zero dRho leads to zero change
		T1_1step_useP(true, 1 * Tesla);
		T1_1step_useP(true, 0.1 * Tesla);
		T1_1step_useP(false, 0.1 * Tesla);
	}
	void relax_rate_usegm(){
		// single-rate calculations
		if (printgm){ // storing all gm matrices may need huge memory; we may not want to print it
			T1_rate_dRho(true, true, false, 0, 0);
			//T1_rate_dRho(true, true, true, 0., 3 * ePhDelta / sqrt(2) / eV);
			//T1_rate_dRho(true, false, false, 0., 0.);
			//T1_rate_dRho(true, false, true, 0., 3 * ePhDelta / sqrt(2) / eV);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with constant smearings:\n");
			logPrintf("**************************************************\n");
			T1_rate_dRho(false, false, false, 0, 0);
			double ePhDelta_forJXU_eV = ePhDelta / sqrt(2) / eV;
			if (nEphDelta * ePhDelta_forJXU_eV / 0.001 > 4) T1_rate_dRho(false, false, true, 0, 0.001);
			if (nEphDelta * ePhDelta_forJXU_eV / 0.002 > 4) T1_rate_dRho(false, false, true, 0, 0.002);
			if (nEphDelta * ePhDelta_forJXU_eV / 0.005 > 4) T1_rate_dRho(false, false, true, 0, 0.005);
			if (nEphDelta * ePhDelta_forJXU_eV / 0.01 > 4) T1_rate_dRho(false, false, true, 0, 0.01);
			//T1_rate_dRho(false, false, true, 0., 0.02);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with ImSigma_eph + a constant:\n");
			logPrintf("**************************************************\n");
			T1_rate_dRho(false, false, true, 1, 0);
			T1_rate_dRho(false, false, true, 0.67, 0);
			T1_rate_dRho(false, false, true, 0.5, 0);
			T1_rate_dRho(false, false, true, 0.33, 0);
			T1_rate_dRho(false, false, true, 0.25, 0);
			if (nEphDelta * ePhDelta_forJXU_eV / 0.001 > 4) T1_rate_dRho(false, false, true, 1, 0.001);
			if (nEphDelta * ePhDelta_forJXU_eV / 0.002 > 4) T1_rate_dRho(false, false, true, 1, 0.002);
			if (nEphDelta * ePhDelta_forJXU_eV / 0.005 > 4) T1_rate_dRho(false, false, true, 1, 0.005);
			if (nEphDelta * ePhDelta_forJXU_eV / 0.01 > 4) T1_rate_dRho(false, false, true, 1, 0.01);
			//T1_rate_dRho(false, false, true, 1., 0.02);
		}
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Initialize sparse matrices for Lindblad dynamics");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- kpoints
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	//--- doping / temperature
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional: lowest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional: highest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double Tmax = inputMap.get("Tmax") * Kelvin; //maximum temperature in Kelvin (ambient phonon T = initial electron T)
	const double nkBT = inputMap.get("nkBT", 7); //energy conservation width for e-ph coupling
	//--- pump
	const double pumpOmegaMax = inputMap.get("pumpOmegaMax") * eV; //maximum pump frequency in eV
	const double pumpTau = inputMap.get("pumpTau") * fs; //maximum pump frequency in eV
	const double probeOmegaMax = inputMap.get("probeOmegaMax") * eV; //maximum probe frequency in eV
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	const bool ePhEnabled = (ePhMode != "Off");
	const bool ePhOnlyElec = inputMap.get("ePhOnlyElec", 0);
	const bool ePhOnlyHole = inputMap.get("ePhOnlyHole", 0);
	bool notBothTrue = !(ePhOnlyElec && ePhOnlyHole); assert(notBothTrue);
	//sqrt(2) for JXU's program
	const double ePhDelta = inputMap.get("ePhDelta", 0.01) * eV * sqrt(2); //energy conservation width for e-ph coupling
	const double nEphDelta = inputMap.get("nEphDelta", nEphDelta_default*sqrt(2)) / sqrt(2); //energy conservation width for e-ph coupling
	const bool detailBalance = inputMap.get("detailBalance", 0);
	const bool variedDelta = inputMap.get("variedDelta", 0);
	const bool antiResonant = inputMap.get("antiResonant", 1);
	const bool printgm = inputMap.get("printgm", true);
	const bool mergegm = inputMap.get("mergegm", false); if (mergegm) assert(printgm);
	const size_t maxNeighbors = inputMap.get("maxNeighbors", 0); //if non-zero: limit neighbors per k by stochastic down-sampling and amplifying the Econserve weights
	const int band_skipped = inputMap.get("band_skipped", -1);
	const int modeStart = inputMap.get("modeStart", 0);
	const int modeStop = inputMap.get("modeStop", -1);
	const bool DEBUG = inputMap.get("DEBUG", 0);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("nkBT = %lg\n", nkBT);
	logPrintf("pumpOmegaMax = %lg\n", pumpOmegaMax);
	logPrintf("pumpTau = %lg\n", pumpTau);
	logPrintf("probeOmegaMax = %lg\n", probeOmegaMax);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("ePhOnlyElec = %d\n", ePhOnlyElec);
	logPrintf("ePhOnlyHole = %d\n", ePhOnlyHole);
	logPrintf("ePhDelta = %lg\n", ePhDelta);
	logPrintf("nEphDelta = %lg\n", nEphDelta);
	logPrintf("detailBalance = %d\n", detailBalance);
	logPrintf("variedDelta = %d\n", variedDelta);
	logPrintf("antiResonant = %d\n", antiResonant);
	logPrintf("printgm = %d\n", printgm);
	logPrintf("mergegm = %d\n", mergegm);
	logPrintf("maxNeighbors = %lu\n", maxNeighbors);
	logPrintf("band_skipped = %d\n", band_skipped);
	logPrintf("modeStart = %d\n", modeStart);
	logPrintf("modeStop = %d\n", modeStop);
	logPrintf("DEBUG = %d\n", DEBUG);

	//Initialize FeynWann:
	FeynWannParams fwp(&inputMap);	fwp.printParams(); // Bext, EzExt and scissor
	fwp.needVelocity = true;
	fwp.needSpin = true;
	fwp.needPhonons = ePhEnabled;
	fwp.maskOptimize = true;
	FeynWann fw(fwp);

	//Construct mesh of k-offsets:
	std::vector<vector3<>> k0;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	if(fw.isTruncated[iDir] && NkMult[iDir]!=1)
		{	logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir+1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkFineInv = inv(Diag(vector3<>(NkFine)));
	vector3<int> ikMult;
	for(ikMult[0]=0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for(ikMult[1]=0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for(ikMult[2]=0; ikMult[2]<NkMult[2]; ikMult[2]++)
		k0.push_back(NkFineInv * ikMult);
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	size_t nKeff = k0.size() * fw.eCountPerOffset() * fw.qOffset.size();
	logPrintf("Effectively sampled nKpts: %lu\n", nKeff);

	//Construct mesh of q-offsets:
	std::vector<vector3<>> q0;
	if (ePhEnabled)
	{
		vector3<int> NqMult;
		for (int iDir = 0; iDir<3; iDir++)
			NqMult[iDir] = NkFine[iDir] / fw.phononSup[iDir];
		vector3<int> iqMult;
		for (iqMult[0] = 0; iqMult[0]<NqMult[0]; iqMult[0]++)
		for (iqMult[1] = 0; iqMult[1]<NqMult[1]; iqMult[1]++)
		for (iqMult[2] = 0; iqMult[2]<NqMult[2]; iqMult[2]++)
			q0.push_back(NkFineInv * iqMult);
	}

	if (ip.dryRun)
	{
		logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Create and initialize lindblad calculator:
	LindbladInit lb(fw, NkFine, dmuMin, dmuMax, Tmax, pumpOmegaMax, pumpTau, probeOmegaMax, ePhEnabled, ePhOnlyElec, ePhOnlyHole, ePhDelta);
	lb.nkBT = nkBT;
	lb.nEphDelta = nEphDelta;
	lb.detailBalance = detailBalance;
	lb.variedDelta = variedDelta;
	lb.antiResonant = antiResonant;
	lb.printgm = printgm;
	lb.mergegm = mergegm;
	lb.band_skipped = band_skipped;
	lb.modeStart = modeStart;
	lb.modeStop = modeStop < 0 ? fw.nModes : modeStop; assert(modeStop <= fw.nModes);
	lb.DEBUG = DEBUG;
	
	//First pass (e only): select k-points
	fw.eEneOnly = true;
	lb.kpointSelect(k0);
	fw.eEneOnly = false;

	//Second pass (ph only): select k pairs
	if(ePhEnabled)
		lb.kpairSelect(q0);

	//Final pass: output electronic and e-ph quantities
	lb.saveData();
	if (ePhEnabled)
		lb.relax_1step_useP();
	lb.merge_eph_P(); // This is safer for mpi output
	if (ePhEnabled)
		lb.relax_rate_usegm();
	lb.merge_eph_gm();
	
	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}
