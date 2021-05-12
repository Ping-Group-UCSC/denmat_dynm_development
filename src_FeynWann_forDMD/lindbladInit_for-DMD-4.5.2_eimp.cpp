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
various band indices;
number of k points in the whole grids (needed in prefactor) and in energy windowns;
eigen energies;
spin, momentum, U matrices;

number of k-k' piars in energy windowns;
kpair Information: (i) indeces of k and k' (ii) P1_kk' and P2_kk'
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
#include "help_lindbladInit_for-DMD.h"
#include <gsl/gsl_cblas.h>
#include "sparse_matrix_for-DMD.h"

//Reverse iterator for pointers:
template<class T> constexpr std::reverse_iterator<T*> reverse(T* i) { return std::reverse_iterator<T*>(i); }

static const double omegabyTCut = 1e-3;
static const double nScattDelta_default = 4.*sqrt(2); //number of scattDelta to include in output
static const double degthr_default = 1e-8;

//Lindblad initialization using FeynWann callback
struct LindbladInit_eimp
{
	bool DEBUG;
	FeynWann& fw;
	bool writeU;
	const vector3<int>& NkFine; //!< effective k-point mesh sampled
	const double nkTot; //!< total k-points effectively used in BZ sampling
	double degthr;

	const double dmuMin, dmuMax, Tmax;
	double dmu, n_dmu;
	double nkBT;
	const double pumpOmegaMax, pumpTau, probeOmegaMax;
	double EBot_set, ETop_set;

	const int iDefect; string siDefect, siDefectHole;
	const double defect_density; double defect_fraction;
	bool ePhEnabled, onlyInterValley, onlyIntraValley, eScattOnlyElec, eScattOnlyHole; //!< whether e-ph coupling is enabled
	string shole;
	bool detailBalance;
	const double scattDelta; double nScattDelta; //!< Gaussian energy conservation width
	bool writeg, mergeg, write_sparseP;
	int band_skipped;
	int modeStart, modeStop;
	bool read_kpts, read_kpairs, kparis_eph_eimp;
	const matrix3<> G, GGT;

	// ?Start and ?Stop are used most frequently and used as global variables in some subroutines
	// ?_probe for probe and can contain bands far from band edges; ?_scatt for scattering; ?_dm for states related to density matrix change
	int bStart, bStop, bCBM, nBandsSel, bBot_scatt, bTop_scatt, bBot_dm, bTop_dm, bBot_probe, bTop_probe, bRef, nBandsSel_probe;
	double EvMax, EcMin, Emid; //VBM and CBM estimates, Emid is the energy in the middle of the gap
	double Estart, Estop, EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_scatt, ETop_scatt; //energy range for k selection

	LindbladInit_eimp(FeynWann& fw, const vector3<int>& NkFine,
		double dmuMin, double dmuMax, double Tmax,
		double pumpOmegaMax, double pumpTau, double probeOmegaMax,
		int iDefect, double defect_density,
		bool ePhEnabled, bool eScattOnlyElec, bool eScattOnlyHole, double scattDelta)
		: fw(fw), NkFine(NkFine), nkTot((double)NkFine[0] * (double)NkFine[1] * (double)NkFine[2]),
		dmuMin(dmuMin), dmuMax(dmuMax), dmu(dmuMax), Tmax(Tmax),
		pumpOmegaMax(pumpOmegaMax), pumpTau(pumpTau), probeOmegaMax(probeOmegaMax),
		iDefect(iDefect), defect_density(defect_density), defect_fraction(defect_density*cell_size(fw)),
		ePhEnabled(ePhEnabled), eScattOnlyElec(eScattOnlyElec), eScattOnlyHole(eScattOnlyHole), shole(eScattOnlyHole ? "_hole" : ""), scattDelta(scattDelta),
		G(2 * M_PI * inv(fw.R)), GGT(G * (~G))
	{
		ostringstream convert; convert << iDefect; convert.flush(); MPI_Barrier(MPI_COMM_WORLD);
		siDefect = "_D" + convert.str(); siDefectHole = siDefect + shole;
		logPrintf("defect_fraction = %lg\n", defect_fraction);
	}

	//--------- k-point selection -------------

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
		((LindbladInit_eimp*)params)->eRange(state);
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
		((LindbladInit_eimp*)params)->bSelect(state);
	}
	void bSelect_driver(const double& EBot, const double& ETop, int& bBot, int& bTop){
		Estart = EBot; Estop = ETop;
		bStart = fw.nBands; bStop = 0; bCBM = 0;
		//fw.eLoop(vector3<>(), LindbladInit::bSelect, this);
		for (vector3<> qOff : fw.qOffset)
			fw.eLoop(qOff, LindbladInit_eimp::bSelect, this);
		mpiWorld->allReduce(bStart, MPIUtil::ReduceMin);
		mpiWorld->allReduce(bStop, MPIUtil::ReduceMax);
		mpiWorld->allReduce(bCBM, MPIUtil::ReduceMax);
		bBot = bStart; bTop = bStop;
	}

	std::vector<vector3<>> k; //selected k-points
	std::vector<double> E; //all band energies for selected k-points
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
		}
	}
	static void kSelect(const FeynWann::StateE& state, void* params){
		((LindbladInit_eimp*)params)->kSelect(state);
	}
	void kpointSelect(const std::vector<vector3<>>& k0){
		//Parallel:
		size_t oStart, oStop; //range of offstes handled by this process group
		if (mpiGroup->isHead()) TaskDivision(k0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart); mpiGroup->bcast(oStop);
		size_t noMine = oStop - oStart;
		size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

		// Determin VBM and CBM
		EvMax = -DBL_MAX; EcMin = +DBL_MAX;
		//fw.eLoop(vector3<>(), LindbladInit::eRange, this);
		for (vector3<> qOff : fw.qOffset) fw.eLoop(qOff, LindbladInit_eimp::eRange, this);
		mpiWorld->allReduce(EvMax, MPIUtil::ReduceMax); mpiWorld->allReduce(EcMin, MPIUtil::ReduceMin);
		Emid = (EvMax + EcMin) / 2.;
		logPrintf("VBM: %.6lf eV, CBM: %.6lf eV, Middle: %.6lf\n", EvMax / eV, EcMin / eV, Emid/eV);
		logPrintf("Note that VBM and CBM may not be determined correctly,\nyou may have to use band_skipped to set starting band index of wannier bands\n");
		if (EvMax < EcMin && band_skipped >= 0) fw.isMetal = false;

		//Determine energy range:
		//--- add margins of max phonon energy, energy conservation width and fermiPrime width
		double Emargin = nkBT*Tmax; //neglect below 10^-3 occupation deviation from equilibrium
		if (ETop_set > EBot_set){
			EBot_dm = EBot_set;
			EBot_scatt = (!fw.isMetal && eScattOnlyElec) ? Emid : EBot_dm;
			ETop_dm = ETop_set;
			ETop_scatt = (!fw.isMetal && eScattOnlyHole) ? Emid : ETop_dm;
		}
		else if (!fw.isMetal){
			logPrintf("The system is not metalic\n");
			double EBot_pump = EcMin - pumpOmegaMax - 5. / pumpTau;
			EBot_scatt = eScattOnlyElec ? Emid : std::min(EBot_pump, std::min(EvMax, dmuMin) - Emargin);
			EBot_dm = std::min(EBot_pump, EBot_scatt);

			double ETop_pump = EvMax + pumpOmegaMax + 5. / pumpTau;
			ETop_scatt = eScattOnlyHole ? Emid : std::max(ETop_pump, std::max(EcMin, dmuMax) + Emargin);
			ETop_dm = std::max(ETop_pump, ETop_scatt);
		}
		else{
			logPrintf("The system is metalic\n");
			EBot_scatt = dmuMin - pumpOmegaMax - std::max(5. / pumpTau + 3.*Tmax, Emargin);
			EBot_dm = EBot_scatt;
			ETop_scatt = dmuMax + pumpOmegaMax + std::max(5. / pumpTau + 3.*Tmax, Emargin);
			ETop_dm = ETop_scatt;
		}
		EBot_probe = EBot_dm - probeOmegaMax - 5. / pumpTau;
		ETop_probe = ETop_dm + probeOmegaMax + 5. / pumpTau;
		logPrintf("Emargin = %.3lf eV\n", Emargin / eV);
		logPrintf("Active energy range for probe: %.3lf to %.3lf eV\n", EBot_probe / eV, ETop_probe / eV);
		logPrintf("Active energy range for kpointSelect: %.3lf to %.3lf eV\n", EBot_scatt / eV, ETop_scatt / eV);
		logPrintf("Active energy range for density matrix: %.3lf to %.3lf eV\n", EBot_dm / eV, ETop_dm / eV);

		//Select bands:
		bSelect_driver(EBot_probe, ETop_probe, bBot_probe, bTop_probe);
		nBandsSel_probe = bTop_probe - bBot_probe; bRef = bBot_probe;
		logPrintf("\nbBot_probe= %d bTop_probe= %d\n", bBot_probe, bTop_probe);
		bSelect_driver(EBot_dm, ETop_dm, bBot_dm, bTop_dm);
		logPrintf("\nbBot_dm= %d bTop_dm= %d\n", bBot_dm, bTop_dm);
		bSelect_driver(EBot_scatt, ETop_scatt, bBot_scatt, bTop_scatt);
		if (fw.isMetal) bCBM = bBot_probe;
		logPrintf("\nbBot_scatt= %d bTop_scatt= %d bCBM= %d\n", bBot_scatt, bTop_scatt, bCBM);

		//Select k-points:
		Estart = EBot_scatt; Estop = ETop_scatt;
		bStart = bBot_scatt; bStop = bTop_scatt; nBandsSel = bStop - bStart;

		if (read_kpts){
			FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
			char s[200]; fgets(s, sizeof s, fp); fgets(s, sizeof s, fp);
			if (fgets(s, sizeof s, fp) != NULL){
				double dtmp; int nk; sscanf(s, "%le %d", &dtmp, &nk);
				k.resize(nk, vector3<>(0, 0, 0)); E.resize(nk*nBandsSel_probe); logPrintf("number of k = %lu\n", k.size());
			}
			fclose(fp);
			FILE *fpk = fopen("ldbd_data/ldbd_kvec.bin", "rb"), *fpe = fopen("ldbd_data/ldbd_ek.bin", "rb");
			size_t expected_size_k = k.size() * 3 * sizeof(double), expected_size_e = k.size() * nBandsSel_probe * sizeof(double);
			check_file_size(fpk, expected_size_k, "ldbd_kvec.bin size does not match expected size"); check_file_size(fpe, expected_size_e, "ldbd_ek.bin size does not match expected size");
			fread(k.data(), sizeof(double), k.size() * 3, fpk);
			fread(E.data(), sizeof(double), k.size() * nBandsSel_probe, fpe);
			fclose(fpk); fclose(fpe);
			return;
		}

		logPrintf("Scanning k-points with active states: "); logFlush();
		for (size_t o = oStart; o<oStop; o++){
			//fw.eLoop(k0[o], LindbladInit::kSelect, this);
			for (vector3<> qOff : fw.qOffset) fw.eLoop(k0[o] + qOff, LindbladInit_eimp::kSelect, this);
			if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); } //Print progress:
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
			std::vector<double> E(nkSelected*nBandsSel_probe); // jxu
			std::copy(this->k.begin(), this->k.end(), k.begin() + nkPrev[mpiWorld->iProcess()]);
			std::copy(this->E.begin(), this->E.end(), E.begin() + nkPrev[mpiWorld->iProcess()] * nBandsSel_probe); // jxu
			//Broadcast:
			for (int jProc = 0; jProc < mpiWorld->nProcesses(); jProc++){
				size_t ikStart = nkPrev[jProc], nk = nkPrev[jProc + 1] - ikStart;
				mpiWorld->bcast(k.data() + ikStart, nk, jProc);
				mpiWorld->bcast(E.data() + ikStart*nBandsSel_probe, nk*nBandsSel_probe, jProc); // jxu
			}
			//Store to class variables:
			std::swap(k, this->k);
			std::swap(E, this->E);
		}
		logPrintf("Found %lu k-points with active states from %21.14le total k-points (%.0fx reduction)\n\n",
			nkSelected, nkTot, round(nkTot*1. / nkSelected));
	}

	//--------- k-pair selection -------------
	std::vector<std::vector<size_t>> kpartners; //list of e-ph coupled k2 for each k1
	std::vector<std::pair<size_t, size_t>> kpairs; //pairs of k1 and k2
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
	inline bool findK(vector3<> k, size_t&ik){
		const std::map<size_t, size_t>::iterator iter = kIndexMap.find(kIndex(k));
		if (iter != kIndexMap.end())
		{
			ik = iter->second;
			return true;
		}
		else return false;
	}
	inline void selectActive(const double*& Ebegin, const double*& Eend, double Elo, double Ehi){ //narrow pointer range to data within [Estart,Estop]
		Ebegin = std::lower_bound(Ebegin, Eend, Elo);
		Eend = &(*std::lower_bound(reverse(Eend), reverse(Ebegin), Ehi, std::greater<double>())) + 1;
	}
	inline void kpSelect(const FeynWann::StatePh& state)
	{	//Find pairs of momentum conserving electron states with this q:
		for (size_t ik1 = 0; ik1 < k.size(); ik1++){
			const vector3<>& k1 = k[ik1];
			vector3<> k2 = k1 - state.q; //momentum conservation
			if (onlyInterValley && !isInterValley(GGT, k1, k2)) continue;
			if (onlyIntraValley && isInterValley(GGT, k1, k2)) continue;
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
					double dE = (*E1) - (*E2);
					if ((writeU || kparis_eph_eimp) && fabs(dE) < nScattDelta*scattDelta)
						Econserve = true;
					else if (modeStop > modeStart){
						for (const double omegaPh : state.omega) if (omegaPh/Tmax>omegabyTCut){ //loop over non-zero phonon frequencies
							double deltaE_minus = dE - omegaPh; //energy conservation violation
							double deltaE_plus = dE + omegaPh; //energy conservation violation
							if (fabs(deltaE_minus) < nScattDelta*scattDelta || fabs(deltaE_plus) < nScattDelta*scattDelta){ //else negligible at the 10^-3 level for a Gaussian
								Econserve = true;
								break;
							}
						}
					}
					if (Econserve) break;
				}
				if (Econserve) break;
			}
			if (Econserve) kpairs.push_back(std::make_pair(ik1, ik2));
		}
	}
	static void kpSelect(const FeynWann::StatePh& state, void* params){
		((LindbladInit_eimp*)params)->kpSelect(state);
	}
	void kpairSelect(const std::vector<vector3<>>& q0){
		logPrintf("\nFor kpairSelect: Estart= %lg Estop= %lg bStart= %d bStop= %d\n\n", Estart, Estop, bStart, bStop); logFlush();
		
		//Initialize kIndexMap for searching selected k-points:
		for (size_t ik = 0; ik < k.size(); ik++) kIndexMap[kIndex(k[ik])] = ik;

		//When we need all k pairs
		if (nScattDelta > 999){
			for (size_t ik1 = 0; ik1 < k.size(); ik1++)
			for (size_t ik2 = ik1; ik2 < k.size(); ik2++)
				kpairs.push_back(std::make_pair(ik1, ik2));
			logPrintf("Number of pairs: %lu\n\n", kpairs.size());

			if (mpiWorld->isHead()) std::random_shuffle(kpairs.begin(), kpairs.end());
			logPrintf("Randomly rearranging kpairs done\n");
			mpiWorld->bcast((size_t*)kpairs.data(), kpairs.size() * 2);
			logPrintf("bcast kpairs done\n");
			return;
		}

		if (read_kpairs){
			FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
			char s[200]; fgets(s, sizeof s, fp); fgets(s, sizeof s, fp); fgets(s, sizeof s, fp);
			if (fgets(s, sizeof s, fp) != NULL){
				int itmp, nkpair;
				if (eScattOnlyHole) sscanf(s, "%d %d", &itmp, &nkpair);
				else sscanf(s, "%d", &nkpair);
				kpairs.resize(nkpair); logPrintf("number of kpairs = %lu\n", kpairs.size());
			}
			fclose(fp);
			string fnamek = dir_ldbd + "ldbd_kpair_k1st" + shole + ".bin"; string fnamekp = dir_ldbd + "ldbd_kpair_k2nd" + shole + ".bin";
			FILE *fpk = fopen(fnamek.c_str(), "rb"), *fpkp = fopen(fnamekp.c_str(), "rb");
			size_t expected_size = kpairs.size() * sizeof(size_t);
			check_file_size(fpk, expected_size, "ldbd_kpair_k1st" + shole + ".bin size does not match expected size");
			check_file_size(fpkp, expected_size, "ldbd_kpair_k2nd" + shole + ".bin size does not match expected size");
			for (size_t ikpair = 0; ikpair < kpairs.size(); ikpair++){
				fread(&kpairs[ikpair].first, sizeof(size_t), 1, fpk);
				fread(&kpairs[ikpair].second, sizeof(size_t), 1, fpkp);
			}
			fclose(fpk); fclose(fpkp);
			return;
		}

		//Parallel:
		size_t oStart, oStop; //!< range of offstes handled by this process groups
		if (mpiGroup->isHead()) TaskDivision(q0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart); mpiGroup->bcast(oStop);
		size_t noMine = oStop - oStart;
		size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

		//Find momentum-conserving k-pairs for which energy conservation is also possible for some bands:
		logPrintf("Scanning k-pairs with e-ph coupling: "); logFlush();
		for (size_t o = oStart; o<oStop; o++){
			fw.phLoop(q0[o], LindbladInit_eimp::kpSelect, this);
			if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();

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
		logPrintf("Found %lu k-pairs with e-ph coupling from %lu total pairs of selected k-points (%.0fx reduction)\n",
			nkpairs, nkpairsTot, round(nkpairsTot*1. / nkpairs));
	}

	//--------- Save data -------------
	string dir_ldbd = "ldbd_data/";
	string dir_debug = "debug_info/";
	void saveR(){
		if (mpiWorld->isHead()){
			//int dim = 3;
			//for (int idir = 0; idir < 3; idir++)
			//	if (fw.isTruncated[idir]) dim -= 1;
			string fname = dir_ldbd + "ldbd_R.dat";
			FILE *fp = fopen(fname.c_str(), "w");
			//fprintf(fp, "%d\n", dim);
			fprintf(fp, "%d\n", dimension(fw));
			fw.R.print(fp, " %14.7le", false);
			fclose(fp);
		}
	}
	void saveSize(){
		if (mpiWorld->isHead()){
			string fname = dir_ldbd + "ldbd_size.dat";
			FILE *fp = fopen(fname.c_str(), "w");
			if (eScattOnlyElec)
				fprintf(fp, "There are scatterings only for conduction electrons\n");
			else if (Estop < Emid)
				fprintf(fp, "There are scatterings only for valence electrons\n");
			else
				fprintf(fp, "There are scatterings for both valence and conduction electrons\n");
			if (eScattOnlyElec)
				fprintf(fp, "%d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_scatt_elec bTop_scatt_elec nb_wannier bBot_probe\n",
				bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, bBot_scatt - bRef, bTop_scatt - bRef, fw.nBands, bRef);
			else if (eScattOnlyHole)
				fprintf(fp, "%d %d %d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_scatt_elec bTop_scatt_elec bBot_scatt_hole bTop_scatt_hole nb_wannier bBot_probe\n",
				bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, 0, 0, bBot_scatt - bRef, bTop_scatt - bRef, fw.nBands, bRef);
			else
				fprintf(fp, "%d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_scatt bTop_scatt nb_wannier bBot_probe\n",
				bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, bBot_scatt - bRef, bTop_scatt - bRef, fw.nBands, bRef);
			fprintf(fp, "%21.14le %lu %d %d %d # nk_full nk kmesh\n", nkTot, k.size(), NkFine[0], NkFine[1], NkFine[2]);
			if (eScattOnlyElec)
				fprintf(fp, "%lu # nkpair_elec\n", kpairs.size());
			else if (eScattOnlyHole)
				fprintf(fp, "%d %lu # nkpair_elec nkpair_hole\n", 0, kpairs.size());
			else
				fprintf(fp, "%lu # nkpair\n", kpairs.size());
			fprintf(fp, "%d %d # modeStart modeStp\n", modeStart, modeStop);
			fprintf(fp, "%21.14le # T\n", Tmax); // Currently T = Tmax
			fprintf(fp, "%21.14le %21.14le %21.14le # muMin, muMax mu (given carrier density)\n", dmuMin, dmuMax, dmu);
			fprintf(fp, "%lg %lg # degauss\n", scattDelta, nScattDelta);
			fprintf(fp, "%14.7le %14.7le %14.7le %14.7le %14.7le %14.7le # EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_scatt, ETop_scatt", EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_scatt, ETop_scatt);
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
			for (size_t ik = tasks.start(whose); ik < tasks.stop(whose); ik++)
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
		if (mpiWorld->isHead() && writeU){
			string fnameu = dir_ldbd + "ldbd_Umat.bin";
			FILE *fpu = fopen(fnameu.c_str(), "wb");
			for (size_t ik = 0; ik < k.size(); ik++){
				matrix U = state_elec[ik].U; // full U matrix
				matrix Ut = transpose(U); // from ColMajor to RowMajor
				fwrite(Ut.data(), 2 * sizeof(double), fw.nBands*fw.nBands, fpu);
			}
			fclose(fpu);
		}
	}
	void saveKpair(){
		if (mpiWorld->isHead()){
			string fnamek = dir_ldbd + "ldbd_kpair_k1st"+shole+".bin"; string fnamekp = dir_ldbd + "ldbd_kpair_k2nd"+shole+".bin";
			FILE *fpk = fopen(fnamek.c_str(), "wb"), *fpkp = fopen(fnamekp.c_str(), "wb");
			for (size_t ikpair = 0; ikpair < kpairs.size(); ikpair++){
				fwrite(&kpairs[ikpair].first, sizeof(size_t), 1, fpk);
				fwrite(&kpairs[ikpair].second, sizeof(size_t), 1, fpkp);
			}
			fclose(fpk); fclose(fpkp);
		}
	}
	
	int ikpairStart, ikpairStop, nkpairMine;
	complex *P1, *P2, *A, *A2;
	double **imsig, **imsigp;
	FILE *fpd;
	
	FILE* fopenP(string fname, string rw){
		fname = dir_ldbd + fname;
		return fopen(fname.c_str(), rw.c_str());
	}
	void fwriteP(FILE *fp1, FILE *fp2, FILE *fp1ns, FILE *fp1s, FILE *fp1i, FILE *fp1j, FILE *fp2ns, FILE *fp2s, FILE *fp2i, FILE *fp2j){
		if (!write_sparseP){
			fwrite(P1, 2 * sizeof(double), (int)std::pow(nBandsSel, 4), fp1); fflush(fp1);
			fwrite(P2, 2 * sizeof(double), (int)std::pow(nBandsSel, 4), fp2); fflush(fp2);
		}
		else{
			sparse_mat *sP1 = new sparse_mat(P1, nBandsSel*nBandsSel);
			sP1->write_to_files(fp1ns, fp1s, fp1i, fp1j); fflush(fp1ns); fflush(fp1s); fflush(fp1i); fflush(fp1j);
			sparse_mat *sP2 = new sparse_mat(P2, nBandsSel*nBandsSel);
			sP2->write_to_files(fp2ns, fp2s, fp2i, fp2j); fflush(fp2ns); fflush(fp2s); fflush(fp2i); fflush(fp2j);
			delete sP1; delete sP2;
		}
	}

	void compute_eimp(){
		ostringstream convert; convert << mpiWorld->iProcess();
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise files are not created for non-root processes
		FILE *fp1, *fp2, *fp1c, *fp2c;
		FILE *fp1ns, *fp2ns, *fp1s, *fp2s, *fp1i, *fp2i, *fp1j, *fp2j, *fp1cns, *fp2cns, *fp1cs, *fp2cs, *fp1ci, *fp2ci, *fp1cj, *fp2cj;
		if (!write_sparseP){
			fp1 = fopenP("ldbd_P1_lindblad" + siDefectHole + ".bin." + convert.str(), "wb"); fp1c = fopenP("ldbd_P1_conventional" + siDefectHole + ".bin." + convert.str(), "wb");
			fp2 = fopenP("ldbd_P2_lindblad" + siDefectHole + ".bin." + convert.str(), "wb"); fp2c = fopenP("ldbd_P2_conventional" + siDefectHole + ".bin." + convert.str(), "wb");
		}
		else{
			fp1ns = fopenP("sP1_lindblad" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp1s = fopenP("sP1_lindblad" + siDefectHole + "_s.bin." + convert.str(), "wb");
			fp1i = fopenP("sP1_lindblad" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp1j = fopenP("sP1_lindblad" + siDefectHole + "_j.bin." + convert.str(), "wb");
			fp2ns = fopenP("sP2_lindblad" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp2s = fopenP("sP2_lindblad" + siDefectHole + "_s.bin." + convert.str(), "wb");
			fp2i = fopenP("sP2_lindblad" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp2j = fopenP("sP2_lindblad" + siDefectHole + "_j.bin." + convert.str(), "wb");
			fp1cns = fopenP("sP1_conventional" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp1cs = fopenP("sP1_conventional" + siDefectHole + "_s.bin." + convert.str(), "wb");
			fp1ci = fopenP("sP1_conventional" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp1cj = fopenP("sP1_conventional" + siDefectHole + "_j.bin." + convert.str(), "wb");
			fp2cns = fopenP("sP2_conventional" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp2cs = fopenP("sP2_conventional" + siDefectHole + "_s.bin." + convert.str(), "wb");
			fp2ci = fopenP("sP2_conventional" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp2cj = fopenP("sP2_conventional" + siDefectHole + "_j.bin." + convert.str(), "wb");
		}
		string fnameg = dir_ldbd + "ldbd_g.bin." + convert.str();
		FILE *fpg; if (writeg) fpg = fopen(fnameg.c_str(), "wb");
		string fnamesig = dir_ldbd + "ldbd_imsig" + siDefect + ".bin";
		FILE *fpsig = fopen(fnamesig.c_str(), "wb");
		bool ldebug = DEBUG;
		string fnamed = dir_debug + "ldbd_debug_compute_eimp.out." + convert.str();
		if (ldebug) fpd = fopen(fnamed.c_str(), "w");

		// the index order is consistent with the file name order
		TaskDivision(kpairs.size(), mpiWorld).myRange(ikpairStart, ikpairStop);
		nkpairMine = ikpairStop - ikpairStart;
		size_t nkpairInterval = std::max(1, int(round(nkpairMine / 50.))); //interval for reporting progress
		MPI_Barrier(MPI_COMM_WORLD);

		A = alloc_array(nBandsSel*nBandsSel); A2 = alloc_array(nBandsSel*nBandsSel);
		P1 = alloc_array((int)std::pow(nBandsSel, 4)); P2 = alloc_array((int)std::pow(nBandsSel, 4));
		imsig = alloc_real_array(k.size(), nBandsSel); imsigp = alloc_real_array(k.size(), nBandsSel);

		logPrintf("Compute E-I: \n"); logFlush();
		if (ldebug){
			fprintf(fpd, "\nikpairStart= %d, ikpairStop= %d\n", ikpairStart, ikpairStop); fflush(fpd);
		}
		for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
			int ikpair_glob = ikpair_local + ikpairStart;
			int ik = kpairs[ikpair_glob].first, jk = kpairs[ikpair_glob].second;
			if (ldebug){
				fprintf(fpd, "\nik= %d, k= %lg %lg %lg, ikp= %d, kp= %lg %lg %lg\n",
					ik, k[ik][0], k[ik][1], k[ik][2], jk, k[jk][0], k[jk][1], k[jk][2]); fflush(fpd);
			}
			diagMatrix Ek = state_elec[ik].E(bStart, bStop), Ekp = state_elec[jk].E(bStart, bStop);
			if (ldebug){
				for (int b = 0; b < nBandsSel; b++){
					fprintf(fpd, "Ek[%d]= %lg Ekp[%d]= %lg\n", b, Ek[b], b, Ekp[b]); fflush(fpd);
				}
			}
			FeynWann::MatrixDefect mD;
			fw.defectCalc(state_elec[ik], state_elec[jk], mD); // g^i_kk'

			if (writeg){
				matrix g = mD.M(bStart, bStop, bStart, bStop);
				fwrite(g.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpg);
				if (ik < jk){
					matrix gji = dagger(mD.M(bStart, bStop, bStart, bStop));
					fwrite(gji.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpg);
				}
			}

			compute_P_eimp(ik, jk, Ek, Ekp, mD, true, ldebug, true, false); // gaussian smearing
			fwriteP(fp1, fp2, fp1ns, fp1s, fp1i, fp1j, fp2ns, fp2s, fp2i, fp2j);
			
			compute_P_eimp(ik, jk, Ek, Ekp, mD, false, ldebug, false, false); // conventional, gaussian smearing
			fwriteP(fp1c, fp2c, fp1cns, fp1cs, fp1ci, fp1cj, fp2cns, fp2cs, fp2ci, fp2cj);

			//Print progress:
			if ((ikpair_local + 1) % nkpairInterval == 0) { logPrintf("%d%% ", int(round((ikpair_local + 1)*100. / nkpairMine))); logFlush(); }
		}
		for (size_t ik = 0; ik < k.size(); ik++){
			mpiWorld->allReduce(&imsig[ik][0], nBandsSel, MPIUtil::ReduceSum);
			mpiWorld->allReduce(&imsigp[ik][0], nBandsSel, MPIUtil::ReduceSum);
		}
		if (mpiWorld->isHead()){
			for (size_t ik = 0; ik < k.size(); ik++)
				fwrite(imsig[ik], sizeof(double), nBandsSel, fpsig);
			write_imsige();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		logPrintf("done.\n"); logFlush();
		if (!write_sparseP){
			fclose(fp1); fclose(fp2); fclose(fp1c); fclose(fp2c);
		}
		else{
			fclose(fp1ns); fclose(fp1s); fclose(fp1i); fclose(fp1j); fclose(fp2ns); fclose(fp2s); fclose(fp2i); fclose(fp2j);
			fclose(fp1cns); fclose(fp1cs); fclose(fp1ci); fclose(fp1cj); fclose(fp2cns); fclose(fp2cs); fclose(fp2ci); fclose(fp2cj);
		}
		if (writeg) fclose(fpg); fclose(fpsig); if (ldebug) fclose(fpd); //if (writeg) fclose(fpwq);
	}

	void compute_P_eimp(int ik, int jk, diagMatrix& Ek, diagMatrix& Ekp, FeynWann::MatrixDefect& mD,
		bool compute_imsig, bool ldebug, bool lindblad, bool lorentzian){
		ldebug = ldebug && lindblad && !lorentzian;
		// compute_imshig should only be true for one of compute_P in subroutine compute_eph
		double ethr = scattDelta * nScattDelta;
		zeros(P1, (int)std::pow(nBandsSel, 4)); zeros(P2, (int)std::pow(nBandsSel, 4));

		double sigma2 = std::pow(scattDelta, 2);
		double prefac_sqrtexp, prefac_sqrtdelta, prefac_exp, prefac_delta, delta;
		prefac_sqrtexp = -0.25 / sigma2; prefac_exp = -0.5 / sigma2;
		if (lorentzian)
			prefac_delta = scattDelta / M_PI;
		else
			prefac_delta = 1. / (scattDelta * sqrt(2.*M_PI));
		prefac_sqrtdelta = sqrt(prefac_delta);
		double prefac_imsig = M_PI / nkTot * prefac_delta;

		matrix g = mD.M(bStart, bStop, bStart, bStop);
		///*
		if (ldebug){
			fprintf(fpd, "g:");  fflush(fpd);
			for (int b1 = 0; b1 < nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel; b2++){
				fprintf(fpd, " (%lg,%lg)", g(b1, b2).real(), g(b1, b2).imag()); fflush(fpd);
			}
			fprintf(fpd, "\n");  fflush(fpd);
		}
		//*/
		bool conserve = false;
		//G = g sqrt(delta(ek - ekp))
		for (int b1 = 0; b1 < nBandsSel; b1++)
		for (int b2 = 0; b2 < nBandsSel; b2++){
			bool inEwind = Ek[b1] >= Estart && Ek[b1] <= Estop && Ekp[b2] >= Estart && Ekp[b2] <= Estop;
			complex G = c0, G2 = c0;
			double dE = fabs(Ek[b1] - Ekp[b2]);
			if (dE < ethr && inEwind){
				conserve = true;
				if (lorentzian)
					delta = 1. / (std::pow(dE, 2) + sigma2);
				else
					delta = exp(prefac_exp*std::pow(dE, 2));
				if (lindblad)
					G2 = prefac_sqrtdelta * g(b1, b2) * sqrt(delta);
				else
					G2 = prefac_delta * g(b1, b2) * delta;

				if (compute_imsig && (ik != jk || b1 != b2)){
					const vector3<>& v1 = state_elec[ik].vVec[b1]; const vector3<>& v2 = state_elec[jk].vVec[b2];
					double cosThetaScatter = dot(v1, v2) / sqrt(std::max(1e-16, v1.length_squared() * v2.length_squared()));
					double dtmp1 = prefac_imsig * g(b1, b2).norm() * delta;
					imsig[ik][b1] += dtmp1; imsigp[ik][b1] += dtmp1 * (1. - cosThetaScatter);
					if (ik < jk){
						double dtmp2 = prefac_imsig * g(b1, b2).norm() * delta;
						imsig[jk][b2] += dtmp2; imsigp[jk][b2] += dtmp2 * (1. - cosThetaScatter);
					}
				}
			}
			G = lindblad ? G2 : g(b1, b2);
			A[b1*nBandsSel + b2] = G;
			A2[b1*nBandsSel + b2] = G2;
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
						// P2_kk'_n1n2n3n4 is just conj(P1_k'k_n1n2n3n4)
						P1[n12 + i3*nBandsSel + i4] += A[i13] * conj(A2[i2*nBandsSel + i4]);
						P2[n12 + i3*nBandsSel + i4] += A[i31] * conj(A2[i4*nBandsSel + i2]);
					}
				}
			}
		} // if (conserve)

		if (ldebug){
			/*
			fprintf(fpd, "\nimsig[ik]:"); fflush(fpd);
			for (int b = 0; b < nBandsSel; b++){
			fprintf(fpd, " %lg", imsig[ik][b]); fflush(fpd);
			}
			fprintf(fpd, "\nimsig[jk]:"); fflush(fpd);
			for (int b = 0; b < nBandsSel; b++){
			fprintf(fpd, " %lg", imsig[jk][b]); fflush(fpd);
			}
			*/
			///*
			fprintf(fpd, "\nP1:\n"); fflush(fpd);
			for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++){
				fprintf(fpd, " (%lg,%lg)", P1[b1*nBandsSel*nBandsSel + b2].real(), P1[b1*nBandsSel*nBandsSel + b2].imag()); fflush(fpd);
			}
			fprintf(fpd, "\nP2:\n"); fflush(fpd);
			for (int b1 = 0; b1 < nBandsSel*nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel*nBandsSel; b2++){
				fprintf(fpd, " (%lg,%lg)", P2[b1*nBandsSel*nBandsSel + b2].real(), P2[b1*nBandsSel*nBandsSel + b2].imag()); fflush(fpd);
			}
			fprintf(fpd, "\n"); fflush(fpd);
			//*/
		}
	}
	void write_imsige(){
		string fnamesigkn = dir_ldbd + "ldbd_imsigkn"+siDefect+".out";
		FILE *fpsigkn = fopen(fnamesigkn.c_str(), "w");
		fprintf(fpsigkn, "E(Har) ImSigma_kn(Har) ImSigmaP(Har)\n");
		double imsig_max = imsig[0][0], imsig_min = imsig[0][0];
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			for (int b = 0; b < nBandsSel; b++){
				fprintf(fpsigkn, "%14.7le %14.7le %14.7le\n", Ek[b], imsig[ik][b], imsigp[ik][b]);
				if (imsig[ik][b] > imsig_max) imsig_max = imsig[ik][b];
				if (imsig[ik][b] < imsig_min) imsig_min = imsig[ik][b];
			}
		}
		logPrintf("\nimsig_min = %lg eV imsig_max = %lg eV\n", imsig_min / eV, imsig_max / eV); logFlush();
		fclose(fpsigkn);

		std::vector<double> imsige(102); std::vector<int> nstate(102);
		double Estart_imsige = Estart, Estop_imsige = Estop;
		if (!fw.isMetal){
			if (eScattOnlyElec) Estart_imsige = minval(state_elec, bCBM, bTop_scatt) - std::min(7., nScattDelta) * scattDelta;
			if (eScattOnlyHole) Estop_imsige = maxval(state_elec, bBot_scatt, bCBM) + std::max(7., nScattDelta) * scattDelta;
			if (eScattOnlyElec || eScattOnlyHole) logPrintf("Active energy range for printing ImSigma(E): %.3lf to %.3lf eV\n", Estart_imsige / eV, Estop_imsige / eV);
		}
		double dE = (Estop_imsige - Estart_imsige) / 100;
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			for (int b = 0; b < nBandsSel; b++){
				int ie = round((Ek[b] - Estart_imsige) / dE);
				if (ie >= 0 && ie <= 101){
					nstate[ie]++;
					imsige[ie] += imsig[ik][b];
				}
			}
		}
		string fnamesige = dir_ldbd + "ldbd_imsige" + siDefect + ".out";
		FILE *fpsige = fopen(fnamesige.c_str(), "w");
		fprintf(fpsige, "E(eV) ImSigma(eV) N_States\n");
		for (int ie = 0; ie < 102; ie++){
			if (nstate[ie] > 0){
				imsige[ie] /= nstate[ie];
				fprintf(fpsige, "%14.7le %14.7le %d\n", (Estart_imsige + ie*dE) / eV, imsige[ie] / eV, nstate[ie]);
			}
		}
		fclose(fpsige);
	}
	void merge_eimp_P_mpi(){
		// This subroutine requires that the index order is consistent with the file name order
		complex ctmp; int itmp;
		if (!write_sparseP){
			merge_files_mpi(dir_ldbd + "ldbd_P1_lindblad" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files_mpi(dir_ldbd + "ldbd_P2_lindblad" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4));
			merge_files_mpi(dir_ldbd + "ldbd_P1_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files_mpi(dir_ldbd + "ldbd_P2_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4));
		}
		else{
			merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_j.bin", itmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_j.bin", itmp, 1);
		}
	}
	void merge_eimp_P(){
		// This subroutine requires that the index order is consistent with the file name order
		complex ctmp; int itmp;
		if (!write_sparseP){
			merge_files(dir_ldbd + "ldbd_P1_lindblad" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files(dir_ldbd + "ldbd_P2_lindblad" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4));
			merge_files(dir_ldbd + "ldbd_P1_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files(dir_ldbd + "ldbd_P2_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4));
		}
		else{
			merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_j.bin", itmp, 1);
			merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_j.bin", itmp, 1);
		}
	}
	void merge_eimp_g(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (writeg && mpiWorld->isHead()){
			logPrintf("Merge g (currently delete):\n");
			for (int i = 0; i < mpiWorld->nProcesses(); i++){
				ostringstream convert; convert << i;
				string fnamegi = dir_ldbd + "ldbd_g.bin." + convert.str();
				remove(fnamegi.c_str());
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	std::vector<diagMatrix> F;
	void savekData(){
		// please never use -G? when running this program
		if (mpiWorld->isHead()) system("mkdir ldbd_data");
		if (DEBUG && mpiWorld->isHead()) system("mkdir debug_info");
		if (!read_kpts){
			saveR(); logPrintf("saveR done\n");
			saveSize(); logPrintf("saveSize done\n");
			saveEk(); logPrintf("saveEk done\n");
		}
		saveElec(); logPrintf("saveElec done\n");

		if (n_dmu == 0 || fw.isMetal)
			dmu = dmuMax;
		else{
			double ncarrier = n_dmu * cell_size(fw) * nkTot;
			dmu = find_mu(ncarrier, Tmax, n_dmu > 0 ? EcMin : EvMax, state_elec, bStart, bCBM, bStop);
		}
		F = computeF(Tmax, dmu, state_elec, bStart, bStop);
		double n_maj_dmuMax = compute_ncarrier(dmuMax < Emid, Tmax, dmuMax, state_elec, bStart, bCBM, bStop) / nkTot / cell_size(fw);
		double n_min_dmuMax = compute_ncarrier(dmuMax >= Emid, Tmax, dmuMax, state_elec, bStart, bCBM, bStop) / nkTot / cell_size(fw);
		double n_maj_dmuMin = compute_ncarrier(dmuMin < Emid, Tmax, dmuMin, state_elec, bStart, bCBM, bStop) / nkTot / cell_size(fw);
		double n_min_dmuMin = compute_ncarrier(dmuMin >= Emid, Tmax, dmuMin, state_elec, bStart, bCBM, bStop) / nkTot / cell_size(fw);
		logPrintf("At dmuMax, majority density: "); print_carrier_density(fw, n_maj_dmuMax);
		logPrintf("At dmuMax, minority density: "); print_carrier_density(fw, n_min_dmuMax);
		logPrintf("At dmuMin, majority density: "); print_carrier_density(fw, n_maj_dmuMin);
		logPrintf("At dmuMin, minority density: "); print_carrier_density(fw, n_min_dmuMin);

		vector3<> bsq_avg = compute_bsq(state_elec, bStart, bStop, degthr, F);
		logPrintf("\nSpin mixing |b|^2: %lg %lg %lg\n", bsq_avg[0], bsq_avg[1], bsq_avg[2]);
	}

	void savekpairData(){
		if (!read_kpairs){
			saveSize(); logPrintf("saveSize done\n");
			saveKpair(); logPrintf("saveKpair done\n");
		}
		logPrintf("\nFor ePh: Estart= %lg eV Estop= %lg eV bStart= %d bStop= %d\n\n", Estart / eV, Estop / eV, bStart, bStop); logFlush();
	}

	//--------- Part 5: Spin relaxation -------------
	complex **dm, **dm1, **ddm, *ddmdt_contrib, *maux1, *maux2;
	complex *P1_next, *P2_next;
	double prefac_scatt;

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
		FILE *fp1, *fp2;
		FILE *fp1ns, *fp2ns, *fp1s, *fp2s, *fp1i, *fp2i, *fp1j, *fp2j;
		string scatt = (lindblad ? "lindblad" : "conventional") + siDefectHole;
		if (!write_sparseP){
			fp1 = fopenP("ldbd_P1_" + scatt + ".bin." + convert.str(), "rb"); fp2 = fopenP("ldbd_P2_" + scatt + ".bin." + convert.str(), "rb");
		}
		else{
			fp1ns = fopenP("sP1_" + scatt + "_ns.bin." + convert.str(), "rb"); fp1s = fopenP("sP1_" + scatt + "_s.bin." + convert.str(), "rb");
			fp1i = fopenP("sP1_" + scatt + "_i.bin." + convert.str(), "rb"); fp1j = fopenP("sP1_" + scatt + "_j.bin." + convert.str(), "rb");
			fp2ns = fopenP("sP2_" + scatt + "_ns.bin." + convert.str(), "rb"); fp2s = fopenP("sP2_" + scatt + "_s.bin." + convert.str(), "rb");
			fp2i = fopenP("sP2_" + scatt + "_i.bin." + convert.str(), "rb"); fp2j = fopenP("sP2_" + scatt + "_j.bin." + convert.str(), "rb");
		}

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

		prefac_scatt = 2 * M_PI / nkTot;
		for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
			int ikpair_glob = ikpair_local + ikpairStart;
			int ik_glob = kpairs[ikpair_glob].first, ikp_glob = kpairs[ikpair_glob].second;

			if (!write_sparseP){
				if (fread(P1, 2 * sizeof(double), std::pow(nBandsSel, 4), fp1) != std::pow(nBandsSel, 4))
					error_message("error during reading P1", "T1_1step_useP");
				if (fread(P2, 2 * sizeof(double), std::pow(nBandsSel, 4), fp2) != std::pow(nBandsSel, 4))
					error_message("error during reading P2", "T1_1step_useP");
			}
			else{
				sparse_mat *sP1 = new sparse_mat(fp1ns, fp1s, fp1i, fp1j);
				sP1->todense(P1, nBandsSel*nBandsSel, nBandsSel*nBandsSel);
				sparse_mat *sP2 = new sparse_mat(fp2ns, fp2s, fp2i, fp2j);
				sP2->todense(P2, nBandsSel*nBandsSel, nBandsSel*nBandsSel);
				delete sP1; delete sP2;
			}
			axbyc(P1, nullptr, std::pow(nBandsSel, 4), c0, defect_fraction);//P1 *= defect_fraction;
			axbyc(P2, nullptr, std::pow(nBandsSel, 4), c0, defect_fraction);//P2 *= defect_fraction;

			compute_ddm(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], P1, P2, ddm[ik_glob]);

			// compute (ikp, ik) contribution
			if (ik_glob < ikp_glob){
				for (int i = 0; i < (int)std::pow(nBandsSel, 4); i++){
					P1_next[i] = conj(P2[i]); P2_next[i] = conj(P1[i]);
				}
				compute_ddm(dm[ikp_glob], dm[ik_glob], dm1[ikp_glob], dm1[ik_glob], P1_next, P2_next, ddm[ikp_glob]);
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
		if (!write_sparseP){ fclose(fp1); fclose(fp2); }
		else{ fclose(fp1ns); fclose(fp1s); fclose(fp1i); fclose(fp1j); fclose(fp2ns); fclose(fp2s); fclose(fp2i); fclose(fp2j); }
	}
	void compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, complex *p1, complex *p2, complex *ddmk){
		zeros(ddmdt_contrib, nBandsSel*nBandsSel);
		term1_P(dm1k, p1, dmkp);
		term2_P(dm1kp, p2, dmk);
		for (int i = 0; i < nBandsSel; i++)
		for (int j = 0; j < nBandsSel; j++)
			ddmk[i*nBandsSel + j] += (prefac_scatt*0.5) * (ddmdt_contrib[i*nBandsSel + j] + conj(ddmdt_contrib[j*nBandsSel + i]));
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

	void T1_rate_dRho(bool drhosdeg, bool obsdeg, bool imsigma_scatt, double fac_imsig, double impurity_broadening){ // impurity_broadening in eV
		ostringstream convert; convert << mpiWorld->iProcess();
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname1 and fname2 are not created for non-root processes
		string fnameg = dir_ldbd + "ldbd_g.bin." + convert.str();
		FILE *fpg = fopen(fnameg.c_str(), "rb");

		std::vector<std::vector<matrix>> Sfull(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		std::vector<std::vector<matrix>> Sdeg(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		std::vector<std::vector<matrix>> dRho(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		for (size_t ik = 0; ik < k.size(); ik++){
			diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			for (int id = 0; id < 3; id++){
				Sfull[ik][id] = state_elec[ik].S[id](bStart, bStop, bStart, bStop);
				degProj(Sfull[ik][id], Ek, degthr, Sdeg[ik][id]);
				if (drhosdeg)
					dRho[ik][id] = dRho_H1(Ek, F[ik], Tmax, Sdeg[ik][id], nBandsSel);
				else
					dRho[ik][id] = dRho_H1(Ek, F[ik], Tmax, Sfull[ik][id], nBandsSel);
			}
		}
		vector3<> dS = compute_spin(dRho, k.size(), nkTot, nBandsSel, bStart, bStop, state_elec);

		vector3<> Sdot(0., 0., 0.);
		double prefac = 2 * M_PI / nkTot / nkTot * defect_fraction;
		double ethr = nScattDelta * scattDelta;

		for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
			int ikpair_glob = ikpair_local + ikpairStart;
			int ik0 = kpairs[ikpair_glob].first, jk0 = kpairs[ikpair_glob].second;

			// we need this trick since g file stores g(jk,ik) right after g(ik,jk) if ik < jk but kpairs array stores only the pair (ik,jk) satisfying ik <= jk
			int nrun = (ik0 == jk0) ? 1 : 2;
			for (int irun = 0; irun < nrun; irun++){
				int ik, jk;
				if (irun == 0){ ik = ik0; jk = jk0; }
				else{ ik = jk0; jk = ik0; }

				diagMatrix Ek = state_elec[ik].E(bStart, bStop), Ekp = state_elec[jk].E(bStart, bStop);

				std::vector<double> prefac_sqrtexp(nBandsSel*nBandsSel), prefac_sqrtdelta(nBandsSel*nBandsSel);
				int bIndex = 0;
				for (int b2 = 0; b2 < nBandsSel; b2++)
				for (int b1 = 0; b1 < nBandsSel; b1++){
					double sigma = fac_imsig * defect_fraction * (imsig[ik][b1] + imsig[jk][b2]) + impurity_broadening * eV;
					sigma = std::max(sigma, 1e-6*eV);
					prefac_sqrtexp[bIndex] = -0.25 / std::pow(sigma, 2);
					prefac_sqrtdelta[bIndex] = 1. / sqrt(sigma * sqrt(2.*M_PI));
					bIndex++;
				}

				vector3<> contrib(0., 0., 0.);
				if (!imsigma_scatt){
					bIndex = 0;
					for (int b2 = 0; b2 < nBandsSel; b2++)
					for (int b1 = 0; b1 < nBandsSel; b1++){
						prefac_sqrtexp[bIndex] = -0.25 / std::pow(scattDelta, 2);
						prefac_sqrtdelta[bIndex] = 1. / sqrt(scattDelta * sqrt(2.*M_PI));
						bIndex++;
					}
				}

				matrix g(nBandsSel, nBandsSel), gtmp(nBandsSel, nBandsSel);
				if (fread(gtmp.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpg) == nBandsSel*nBandsSel){}
				else { error_message("error during reading g", "T1_rate_dRho"); }

				if (irun == 0) g = gtmp;
				else g = dagger(gtmp);

				matrix G(nBandsSel, nBandsSel); complex *GData = G.data();
				bool conserve = false;
				bIndex = 0;
				for (int b2 = 0; b2 < nBandsSel; b2++)
				for (int b1 = 0; b1 < nBandsSel; b1++){
					double dE = Ek[b1] - Ekp[b2];
					if (fabs(dE) < ethr){
						conserve = true;
						double sqrtdelta = exp(prefac_sqrtexp[bIndex] * std::pow(dE, 2));
						*GData = prefac_sqrtdelta[bIndex] * g(b1, b2) * sqrtdelta;
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

				diagMatrix Fjbar(nBandsSel);
				for (int b = 0; b < nBandsSel; b++)
					Fjbar[b] = 1. - F[jk][b];
				std::vector<matrix> dRhoGcomm(3, matrix(nBandsSel, nBandsSel));
				for (int id = 0; id < 3; id++)
					dRhoGcomm[id] = dRho[ik][id] * G * Fjbar - F[ik] * G * dRho[jk][id];

				for (int id = 0; id < 3; id++)
				for (int b2 = 0; b2 < nBandsSel; b2++)
				for (int b1 = 0; b1 < nBandsSel; b1++)
					contrib[id] -= (SGcomm[id](b1, b2).conj() * dRhoGcomm[id](b1, b2)).real();
				Sdot += prefac * contrib;
			}
		}

		mpiWorld->allReduce(Sdot, MPIUtil::ReduceSum);

		vector3<> T1;
		for (int id = 0; id < 3; id++){ T1[id] = -dS[id] / Sdot[id]; }
		const double ps = 1e3*fs; //picosecond
		logPrintf("\ndrhosdeg = %d obsdeg = %d imsigma_scatt = %d fac_imsig = %lg impurity_broadening = %lg eV\n", drhosdeg, obsdeg, imsigma_scatt, fac_imsig, impurity_broadening);
		logPrintf("dS[2] = %lg Sdot[2] = %lg T1 = %lg %lg %lg ps\n", dS[2], Sdot[2], T1[0] / ps, T1[1] / ps, T1[2] / ps); logFlush();
		fclose(fpg);
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
	void relax_rate_useg(){
		// single-rate calculations
		if (writeg){ // storing all g matrices may need huge memory; we may not want to print it
			T1_rate_dRho(true, true, false, 0, 0);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with constant smearings:\n");
			logPrintf("**************************************************\n");
			if (nScattDelta * scattDelta / eV / 0.001 > 4) T1_rate_dRho(false, false, true, 0, 0.001);
			if (nScattDelta * scattDelta / eV / 0.002 > 4) T1_rate_dRho(false, false, true, 0, 0.002);
			if (nScattDelta * scattDelta / eV / 0.005 > 4) T1_rate_dRho(false, false, true, 0, 0.005);
			if (nScattDelta * scattDelta / eV / 0.01 > 4) T1_rate_dRho(false, false, true, 0, 0.01);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with ImSigma_eph + a constant:\n");
			logPrintf("**************************************************\n");
			T1_rate_dRho(false, false, true, 1, 0);
			T1_rate_dRho(false, false, true, 0.67, 0);
			T1_rate_dRho(false, false, true, 0.5, 0);
			T1_rate_dRho(false, false, true, 0.33, 0);
			T1_rate_dRho(false, false, true, 0.25, 0);
			if (nScattDelta * scattDelta / eV / 0.001 > 4) T1_rate_dRho(false, false, true, 1, 0.001);
			if (nScattDelta * scattDelta / eV / 0.002 > 4) T1_rate_dRho(false, false, true, 1, 0.002);
			if (nScattDelta * scattDelta / eV / 0.005 > 4) T1_rate_dRho(false, false, true, 1, 0.005);
			if (nScattDelta * scattDelta / eV / 0.01 > 4) T1_rate_dRho(false, false, true, 1, 0.01);
		}
	}
};

int main(int argc, char** argv)
{
	InitParams ip = FeynWann::initialize(argc, argv, "Initialize electron-impurity matrices for Lindblad dynamics");

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
	const string defectName = inputMap.getString("defectName");
	const int iDefect = inputMap.get("iDefect", 1); // index of defect
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	const bool ePhEnabled = (ePhMode != "Off");
	const bool eScattOnlyElec = inputMap.get("eScattOnlyElec", 0);
	const bool eScattOnlyHole = inputMap.get("eScattOnlyHole", 0);
	bool notBothTrue = !(eScattOnlyElec && eScattOnlyHole); assert(notBothTrue);
	//sqrt(2) for JXU's program
	const bool detailBalance = inputMap.get("detailBalance", 0);
	const double scattDelta = inputMap.get("scattDelta", 0.01) * eV; //energy conservation width for e-ph coupling
	const double nScattDelta = inputMap.get("nScattDelta", nScattDelta_default); //energy conservation width for e-ph coupling
	const bool writeg = inputMap.get("writeg", true);
	const bool mergeg = inputMap.get("mergeg", false); if (mergeg) assert(writeg);
	const size_t maxNeighbors = inputMap.get("maxNeighbors", 0); //if non-zero: limit neighbors per k by stochastic down-sampling and amplifying the Econserve weights
	const int band_skipped = inputMap.get("band_skipped", -1);
	const int modeStart = inputMap.get("modeStart", 0);
	const int modeStop = inputMap.get("modeStop", -1);
	const bool DEBUG = inputMap.get("DEBUG", 0);
	const bool writeU = inputMap.get("writeU", 0); // for electron-impurity and electron-electron scattering models
	double carrier_density = inputMap.get("carrier_density", 0); // unit can be 1, cm-1, cm-2 or cm-3 and is undetermined until dimension of the system has been known
	double defect_density = inputMap.get("defect_density", 0); // unit can be 1, cm-1, cm-2 or cm-3 and is undetermined until dimension of the system has been known
	bool write_sparseP = inputMap.get("write_sparseP", 0);
	double degthr = inputMap.get("degthr", degthr_default);
	double EBot_set = inputMap.get("EBot_set", 0.) * eV;
	double ETop_set = inputMap.get("ETop_set", -1) * eV;
	bool read_kpts = inputMap.get("read_kpts", 1);
	bool read_kpairs = inputMap.get("read_kpairs", 1);
	bool kparis_eph_eimp = inputMap.get("kparis_eph_eimp", 1);
	bool onlyInterValley = inputMap.get("onlyInterValley", 0);
	bool onlyIntraValley = inputMap.get("onlyIntraValley", 0);
	assert(!onlyInterValley || !onlyIntraValley);

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
	logPrintf("defectName = %s\n", defectName.c_str());
	logPrintf("iDefect = %d\n", iDefect);
	logPrintf("eScattOnlyElec = %d\n", eScattOnlyElec);
	logPrintf("eScattOnlyHole = %d\n", eScattOnlyHole);
	logPrintf("detailBalance = %d\n", detailBalance);
	logPrintf("scattDelta = %lg\n", scattDelta);
	logPrintf("nScattDelta = %lg\n", nScattDelta);
	logPrintf("writeg = %d\n", writeg);
	logPrintf("mergeg = %d\n", mergeg);
	logPrintf("maxNeighbors = %lu\n", maxNeighbors);
	logPrintf("band_skipped = %d\n", band_skipped);
	logPrintf("modeStart = %d\n", modeStart);
	logPrintf("modeStop = %d\n", modeStop);
	logPrintf("DEBUG = %d\n", DEBUG);
	logPrintf("writeU = %d\n", writeU);
	logPrintf("write_sparseP = %d\n", write_sparseP);
	logPrintf("degthr = %lg\n", degthr);
	logPrintf("EBot_set = %lg\n", EBot_set);
	logPrintf("ETop_set = %lg\n", ETop_set);
	logPrintf("read_kpts = %d\n", read_kpts);
	logPrintf("read_kpairs = %d\n", read_kpairs);
	logPrintf("kparis_eph_eimp = %d\n", kparis_eph_eimp);
	logPrintf("onlyInterValley = %d\n", onlyInterValley);
	logPrintf("onlyIntraValley = %d\n", onlyIntraValley);

	//Initialize FeynWann:
	FeynWannParams fwp(&inputMap);	fwp.printParams(); // Bext, EzExt and scissor
	fwp.needVelocity = true;
	fwp.needSpin = true;
	fwp.needDefect = defectName;
	fwp.needPhonons = ePhEnabled;
	//fwp.maskOptimize = true;
	FeynWann fw(fwp);

	carrier_density *= cminvdim2au(fw);
	logPrintf("carrier_density = %lg\n", carrier_density);
	defect_density *= cminvdim2au(fw);
	logPrintf("defect_density = %lg\n", defect_density);

	//Construct mesh of k-offsets:
	std::vector<vector3<>> k0;
	vector3<int> NkFine;
	for (int iDir = 0; iDir<3; iDir++){
		if (fw.isTruncated[iDir] && NkMult[iDir] != 1){
			logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir + 1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkFineInv = inv(Diag(vector3<>(NkFine)));
	vector3<int> ikMult;
	for (ikMult[0] = 0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for (ikMult[1] = 0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for (ikMult[2] = 0; ikMult[2]<NkMult[2]; ikMult[2]++)
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
	LindbladInit_eimp lb(fw, NkFine, dmuMin, dmuMax, Tmax, pumpOmegaMax, pumpTau, probeOmegaMax, iDefect, defect_density, ePhEnabled, eScattOnlyElec, eScattOnlyHole, scattDelta);
	lb.nkBT = nkBT;
	lb.detailBalance = detailBalance;
	lb.nScattDelta = nScattDelta;
	lb.writeg = writeg; lb.mergeg = mergeg;
	lb.writeU = writeU;
	lb.band_skipped = band_skipped;
	lb.modeStart = modeStart; lb.modeStop = modeStop < 0 ? fw.nModes : modeStop; assert(modeStop <= fw.nModes);
	logPrintf("modeStop = %d\n", lb.modeStop);
	lb.DEBUG = DEBUG;
	lb.n_dmu = carrier_density;
	lb.write_sparseP = write_sparseP;
	lb.degthr = degthr;
	lb.EBot_set = EBot_set;	lb.ETop_set = ETop_set;
	lb.read_kpts = read_kpts; lb.read_kpairs = read_kpairs; lb.kparis_eph_eimp = kparis_eph_eimp;
	lb.onlyInterValley = onlyInterValley; lb.onlyIntraValley = onlyIntraValley;

	//First pass (e only): select k-points and output electronic quantities
	fw.eEneOnly = true;
	lb.kpointSelect(k0);
	fw.eEneOnly = false;
	lb.savekData();

	//Second pass (ph only): select and output k pairs
	if (read_kpairs || ePhEnabled || nScattDelta > 999){
		lb.kpairSelect(q0);
		lb.savekpairData();
	}

	//Final pass: output e-ph quantities
	if (fwp.needDefect.length()){
		lb.compute_eimp(); // save scattering matrices

		lb.relax_1step_useP();
		//lb.merge_eimp_P(); // This is safer for mpi output
		lb.merge_eimp_P_mpi();
		lb.relax_rate_useg();
		lb.merge_eimp_g();
	}

	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}