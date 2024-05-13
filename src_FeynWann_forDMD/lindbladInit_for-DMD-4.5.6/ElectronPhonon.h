#pragma once
#include "common_headers.h"
#include "parameters.h"
#include "lattice.h"
#include "electron.h"
#include "phonon.h"
#include "sparse_matrix.h"

//Reverse iterator for pointers:
template<class T> constexpr std::reverse_iterator<T*> reverse(T* i) { return std::reverse_iterator<T*>(i); }

static const size_t ngrid = 200;

//Lindblad initialization using FeynWann callback
class ElectronPhonon{
public:
	FeynWann& fw;
	parameters *param;
	lattice *latt;
	electron *elec;
	phonon *ph;

	double Tmax;

	bool read_kpairs, kparis_eph_eimp;

	bool onlyInterValley, onlyIntraValley, ePhOnlyElec, ePhOnlyHole; //!< whether e-ph coupling is enabled
	int modeStart, modeStop, modeSkipStart, modeSkipStop;

	bool detailBalance;
	double ePhDelta, nEphDelta, omegabyTCut; //!< Gaussian energy conservation width
	const double scattDelta; double nScattDelta; //!< Gaussian energy conservation width
	bool needConventional, writegm, keepgm, mergegm, write_sparseP;

	double degthr;

	// ?Start and ?Stop are used most frequently and used as global variables in some subroutines
	// ?_probe for probe and can contain bands far from band edges; ?_eph for e-ph scattering; ?_dm for states related to density matrix change
	int bStart, bStop, bCBM, nBandsSel, nBandsSel_probe;
	double Estart, Estop; //energy range for k selection

	std::vector<vector3<>>& k; //selected k-points
	std::vector<double>& E; //all band energies for selected k-points
	std::vector<diagMatrix>& F;

	// matrix element analysis
	bool analyse_g2_E1E2fix, analyse_g2w;
	double E1fix, E2fix;

	ElectronPhonon(FeynWann& fw, lattice *latt, parameters *param, electron  *elec, phonon *ph)
		: fw(fw), latt(latt), param(param), elec(elec), ph(ph),
		Tmax(param->Tmax), kparis_eph_eimp(param->kparis_eph_eimp), read_kpairs(param->read_kpairs),
		onlyInterValley(param->onlyInterValley), onlyIntraValley(param->onlyIntraValley), ePhOnlyElec(param->ePhOnlyElec), ePhOnlyHole(param->ePhOnlyHole),
		modeStart(param->modeStart), modeStop(param->modeStop), modeSkipStart(param->modeSkipStart), modeSkipStop(param->modeSkipStop),
		detailBalance(param->detailBalance),
		ePhDelta(param->ePhDelta), nEphDelta(param->nEphDelta), omegabyTCut(param->omegabyTCut), degthr(param->degthr),
		scattDelta(param->scattDelta), nScattDelta(param->nScattDelta),
		needConventional(param->needConventional), writegm(param->writegm), keepgm(param->keepgm), mergegm(param->mergegm), write_sparseP(param->write_sparseP),
		k(elec->k), E(elec->E), F(elec->F),
		bStart(elec->bStart), bStop(elec->bStop), bCBM(elec->bCBM), nBandsSel(elec->nBandsSel), nBandsSel_probe(elec->nBandsSel_probe),
		Estart(elec->Estart), Estop(elec->Estop),
		analyse_g2_E1E2fix(elec->analyse_g2_E1E2fix), E1fix(elec->E1fix), E2fix(elec->E2fix)
	{}

	//--------- k-pair selection -------------
	std::vector<std::vector<size_t>> kpartners; //list of e-ph coupled k2 for each k1
	std::vector<std::pair<size_t, size_t>> kpairs; //pairs of k1 and k2
	std::map<size_t, size_t> kIndexMap; //map from k-point mesh index to index in selected set
	inline size_t kIndex(vector3<> k){
		size_t index = 0;
		for (int iDir = 0; iDir < 3; iDir++){
			double ki = k[iDir] - floor(k[iDir]); //wrapped to [0,1)
			index = (size_t)round(elec->NkFine[iDir] * (index + ki));
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
			if (onlyInterValley && !latt->isInterValley(k1, k2)) continue;
			if (onlyIntraValley && latt->isInterValley(k1, k2)) continue;
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
					if ((elec->writeU || kparis_eph_eimp) && fabs(dE) < nScattDelta*scattDelta)
						Econserve = true;
					else if (modeStop > modeStart){
						//for (const double omegaPh : state.omega) if (omegaPh / Tmax > omegabyTCut){
						for (int alpha = modeStart; alpha < modeStop; alpha++){
							if (alpha < modeSkipStop && alpha >= modeSkipStart) continue;
							if (state.omega[alpha] / Tmax > omegabyTCut){ //loop over non-zero phonon frequencies
								double deltaE_minus = dE - state.omega[alpha]; //energy conservation violation
								double deltaE_plus = dE + state.omega[alpha]; //energy conservation violation
								if (fabs(deltaE_minus) < nEphDelta*ePhDelta || fabs(deltaE_plus) < nEphDelta*ePhDelta){ //else negligible at the 10^-3 level for a Gaussian
									Econserve = true;
									break;
								}
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
		((ElectronPhonon*)params)->kpSelect(state);
	}
	void kpairSelect(const std::vector<vector3<>>& q0){
		logPrintf("\nFor kpairSelect: Estart= %lg Estop= %lg bStart= %d bStop= %d\n\n", Estart, Estop, bStart, bStop); logFlush();

		//Initialize kIndexMap for searching selected k-points:
		for (size_t ik = 0; ik < k.size(); ik++) kIndexMap[kIndex(k[ik])] = ik;

		if (read_kpairs){
			logPrintf("\nRead k paris from files.\n"); logFlush();
			assert(fileSize("ldbd_data/ldbd_size.dat") > 0);
			FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
			char s[200]; fgets(s, sizeof s, fp); fgets(s, sizeof s, fp); fgets(s, sizeof s, fp);
			if (fgets(s, sizeof s, fp) != NULL){
				int itmp, nkpair;
				if (ePhOnlyHole) sscanf(s, "%d %d", &itmp, &nkpair);
				else sscanf(s, "%d", &nkpair);
				kpairs.resize(nkpair); logPrintf("number of kpairs = %lu\n", kpairs.size());
			}
			fclose(fp);
			string fnamek = dir_ldbd + "ldbd_kpair_k1st" + shole + ".bin"; string fnamekp = dir_ldbd + "ldbd_kpair_k2nd" + shole + ".bin";
			assert(fileSize(fnamek.c_str()) > 0 && fileSize(fnamekp.c_str()) > 0);
			FILE *fpk = fopen(fnamek.c_str(), "rb"), *fpkp = fopen(fnamekp.c_str(), "rb");
			size_t expected_size = kpairs.size() * sizeof(size_t);
			check_file_size(fpk, expected_size, "ldbd_kpair_k1st" + shole + ".bin size does not match expected size"); check_file_size(fpkp, expected_size, "ldbd_kpair_k2nd" + shole + ".bin size does not match expected size");
			for (size_t ikpair = 0; ikpair < kpairs.size(); ikpair++){
				fread(&kpairs[ikpair].first, sizeof(size_t), 1, fpk);
				fread(&kpairs[ikpair].second, sizeof(size_t), 1, fpkp);
			}
			fclose(fpk); fclose(fpkp);
			return;
		}

		//When we need all k pairs
		if (nEphDelta > 999){
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

		// for matrix element analysis
		if (analyse_g2_E1E2fix){
			std::vector<bool> atE1(k.size(), false), atE2(k.size(), false);
			size_t nk_atE1 = 0, nk_atE2 = 0;
			for (size_t ik = 0; ik < k.size(); ik++){
				for (int b = bStart; b < bStop; b++)
				if (fabs(E[ik*nBandsSel_probe + b - elec->bRef] - E1fix) < ePhDelta * nEphDelta){
					atE1[ik] = true; nk_atE1++; break;
				}
				for (int b = bStart; b < bStop; b++)
				if (fabs(E[ik*nBandsSel_probe + b - elec->bRef] - E2fix) < ePhDelta * nEphDelta){
					atE2[ik] = true; nk_atE2++; break;
				}
			}
			logPrintf("number of k around E1fix: %lu\n", nk_atE1);
			logPrintf("number of k around E2fix: %lu\n", nk_atE2);

			for (size_t ik1 = 0; ik1 < k.size(); ik1++){
				if (!atE1[ik1]) continue;
				for (size_t ik2 = 0; ik2 < k.size(); ik2++)
				if (ik2 != ik1 && atE2[ik2]) kpairs.push_back(std::make_pair(ik1, ik2));
			}
			logPrintf("Number of pairs: %lu\n\n", kpairs.size());
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
		for (size_t o = oStart; o < oStop; o++){
			fw.phLoop(q0[o], ElectronPhonon::kpSelect, this);
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
	complex *P1, *P2, *App, *Apm, *Amp, *Amm, *A2pp, *A2pm, *A2mp, *A2mm;
	double **imsig, **imsigflip, **imsigp;
	
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

	void construct_stateE_for_eph(int ik, const FeynWann::StateE& e, FeynWann::StateE& eout){
		fw.copy_stateE(e, eout);
		eout.dHePhSum.init(fw.nBands*fw.nBands, 3);
		//read dHePhSum from file
		string fname = "ldbd_data/dHePhSum/k" + int2str(ik) + ".bin";
		FILE *fp = fopen(fname.c_str(), "rb");
		if (fread(eout.dHePhSum.data(), 2 * sizeof(double), fw.nBands*fw.nBands * 3, fp) != fw.nBands*fw.nBands * 3)
			error_message("error during reading dHePhSum of ik " + int2str(ik), "construct_stateE_for_eph");
		fclose(fp);
	}

	void compute_eph(){
		ostringstream convert; convert << mpiWorld->iProcess();
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise files are not created for non-root processes
		FILE *fp1, *fp2, *fp1c, *fp2c;
		FILE *fp1ns, *fp2ns, *fp1s, *fp2s, *fp1i, *fp2i, *fp1j, *fp2j, *fp1cns, *fp2cns, *fp1cs, *fp2cs, *fp1ci, *fp2ci, *fp1cj, *fp2cj;
		if (!write_sparseP){
			fp1 = fopenP("ldbd_P1_lindblad" + shole + ".bin." + convert.str(), "wb"); fp2 = fopenP("ldbd_P2_lindblad" + shole + ".bin." + convert.str(), "wb");
			if (needConventional) { fp1c = fopenP("ldbd_P1_conventional" + shole + ".bin." + convert.str(), "wb"); fp2c = fopenP("ldbd_P2_conventional" + shole + ".bin." + convert.str(), "wb"); }
		}
		else{
			fp1ns = fopenP("sP1_lindblad" + shole + "_ns.bin." + convert.str(), "wb"); fp1s = fopenP("sP1_lindblad" + shole + "_s.bin." + convert.str(), "wb");
			fp1i = fopenP("sP1_lindblad" + shole + "_i.bin." + convert.str(), "wb"); fp1j = fopenP("sP1_lindblad" + shole + "_j.bin." + convert.str(), "wb");
			fp2ns = fopenP("sP2_lindblad" + shole + "_ns.bin." + convert.str(), "wb"); fp2s = fopenP("sP2_lindblad" + shole + "_s.bin." + convert.str(), "wb");
			fp2i = fopenP("sP2_lindblad" + shole + "_i.bin." + convert.str(), "wb"); fp2j = fopenP("sP2_lindblad" + shole + "_j.bin." + convert.str(), "wb");
			if (needConventional){
				fp1cns = fopenP("sP1_conventional" + shole + "_ns.bin." + convert.str(), "wb"); fp1cs = fopenP("sP1_conventional" + shole + "_s.bin." + convert.str(), "wb");
				fp1ci = fopenP("sP1_conventional" + shole + "_i.bin." + convert.str(), "wb"); fp1cj = fopenP("sP1_conventional" + shole + "_j.bin." + convert.str(), "wb");
				fp2cns = fopenP("sP2_conventional" + shole + "_ns.bin." + convert.str(), "wb"); fp2cs = fopenP("sP2_conventional" + shole + "_s.bin." + convert.str(), "wb");
				fp2ci = fopenP("sP2_conventional" + shole + "_i.bin." + convert.str(), "wb"); fp2cj = fopenP("sP2_conventional" + shole + "_j.bin." + convert.str(), "wb");
			}
		}
		string fnamegm = dir_ldbd + "ldbd_gm.bin." + convert.str();
		FILE *fpgm; if (writegm) fpgm = fopen(fnamegm.c_str(), "wb");
		string fnamewq = dir_ldbd + "ldbd_wq_kpair.bin." + convert.str(); // phonon frequency of each pair (k,k')
		FILE *fpwq; if (writegm) fpwq = fopen(fnamewq.c_str(), "wb");
		string fnamesig = dir_ldbd + "ldbd_imsig.bin";
		FILE *fpsig = fopen(fnamesig.c_str(), "wb");
		bool ldebug = DEBUG;
		string fnamed = dir_debug + "ldbd_debug_compute_eph.out." + convert.str();
		if (ldebug) fpd = fopen(fnamed.c_str(), "w");

		// the index order is consistent with the file name order
		TaskDivision(kpairs.size(), mpiWorld).myRange(ikpairStart, ikpairStop);
		nkpairMine = ikpairStop - ikpairStart;
		size_t nkpairInterval = std::max(1, int(round(nkpairMine / 50.))); //interval for reporting progress
		MPI_Barrier(MPI_COMM_WORLD);

		App = alloc_array(nBandsSel*nBandsSel); Apm = alloc_array(nBandsSel*nBandsSel); Amp = alloc_array(nBandsSel*nBandsSel); Amm = alloc_array(nBandsSel*nBandsSel);
		A2pp = alloc_array(nBandsSel*nBandsSel); A2pm = alloc_array(nBandsSel*nBandsSel); A2mp = alloc_array(nBandsSel*nBandsSel); A2mm = alloc_array(nBandsSel*nBandsSel);
		P1 = alloc_array((int)std::pow(nBandsSel, 4)); P2 = alloc_array((int)std::pow(nBandsSel, 4));
		imsig = alloc_real_array(k.size(), nBandsSel); imsigflip = alloc_real_array(k.size(), nBandsSel); imsigp = alloc_real_array(k.size(), nBandsSel);

		logPrintf("Compute EPH: \n"); logFlush();
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
			if (onlyInterValley && !latt->isInterValley(k[ik], k[jk])) continue;
			if (onlyIntraValley && latt->isInterValley(k[ik], k[jk])) continue;
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop), Ekp = elec->state_elec[jk].E(bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef), Ekp = elec->state_elec[jk].E(bStart - elec->bRef, bStop - elec->bRef);
			if (ldebug){
				for (int b = 0; b < nBandsSel; b++){
					fprintf(fpd, "Ek[%d]= %lg Ekp[%d]= %lg\n", b, Ek[b], b, Ekp[b]); fflush(fpd);
				}
			}
			FeynWann::StatePh php, phm, php_eph, phm_eph;
			if (modeStop > modeStart) fw.phCalc(k[jk] - k[ik], php); // q = k' -k
			if (modeStop > modeStart) fw.phCalc(k[ik] - k[jk], phm); // q = k - k'
			FeynWann::MatrixEph mp, mm;
			if (modeStop > modeStart){
				if (!elec->save_dHePhSum_disk){
					fw.ePhCalc(elec->state_elec[jk], elec->state_elec[ik], php, mp); // g^-_k'k
					fw.ePhCalc(elec->state_elec[ik], elec->state_elec[jk], phm, mm); // g^-_kk'
				}
				else{// ei and ej will be free after "}"
					FeynWann::StateE ei, ej;
					construct_stateE_for_eph(ik, elec->state_elec[ik], ei);
					construct_stateE_for_eph(jk, elec->state_elec[jk], ej);
					fw.ePhCalc(ej, ei, php, mp); // g^-_k'k
					fw.ePhCalc(ei, ej, phm, mm); // g^-_kk'
				}
			}

			if (modeStop > modeStart && writegm){
				for (int alpha = modeStart; alpha < modeStop; alpha++){
					if (alpha < modeSkipStop && alpha >= modeSkipStart) continue;
					//matrix gm = mm.M[alpha](bStart, bStop, bStart, bStop);
					matrix gm = mm.M[alpha];
					if (phm.omega[alpha]/Tmax <= omegabyTCut) zeros(gm);
					fwrite(gm.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpgm);
					fwrite(&phm.omega[alpha], sizeof(double), 1, fpwq);
				}
				if (ik < jk){
					for (int alpha = modeStart; alpha < modeStop; alpha++){
						if (alpha < modeSkipStop && alpha >= modeSkipStart) continue;
						//matrix gp = dagger(mp.M[alpha](bStart, bStop, bStart, bStop));
						matrix gp = dagger(mp.M[alpha]);
						if (php.omega[alpha]/Tmax <= omegabyTCut) zeros(gp);
						fwrite(gp.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpgm);
						fwrite(&php.omega[alpha], sizeof(double), 1, fpwq);
					}
				}
			}

			compute_P(ik, jk, Ek, Ekp, php, phm, mp, mm, true, ldebug, true, false); // gaussian smearing
			fwriteP(fp1, fp2, fp1ns, fp1s, fp1i, fp1j, fp2ns, fp2s, fp2i, fp2j);
			
			if (needConventional){
				compute_P(ik, jk, Ek, Ekp, php, phm, mp, mm, false, ldebug, false, false); // conventional, gaussian smearing
				fwriteP(fp1c, fp2c, fp1cns, fp1cs, fp1ci, fp1cj, fp2cns, fp2cs, fp2ci, fp2cj);
			}

			//Print progress:
			if ((ikpair_local + 1) % nkpairInterval == 0) { logPrintf("%d%% ", int(round((ikpair_local + 1)*100. / nkpairMine))); logFlush(); }
		}
		for (size_t ik = 0; ik < k.size(); ik++){
			mpiWorld->allReduce(&imsig[ik][0], nBandsSel, MPIUtil::ReduceSum);
			mpiWorld->allReduce(&imsigflip[ik][0], nBandsSel, MPIUtil::ReduceSum);
			mpiWorld->allReduce(&imsigp[ik][0], nBandsSel, MPIUtil::ReduceSum);
		}
		if (mpiWorld->isHead()){
			for (size_t ik = 0; ik < k.size(); ik++)
				fwrite(imsig[ik], sizeof(double), nBandsSel, fpsig);
			write_imsige();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		logPrintf("done.\n\n"); logFlush();
		if (!write_sparseP){
			fclose(fp1); fclose(fp2); if (needConventional){ fclose(fp1c); fclose(fp2c); }
		}
		else{
			fclose(fp1ns); fclose(fp1s); fclose(fp1i); fclose(fp1j); fclose(fp2ns); fclose(fp2s); fclose(fp2i); fclose(fp2j);
			if (needConventional) { fclose(fp1cns); fclose(fp1cs); fclose(fp1ci); fclose(fp1cj); fclose(fp2cns); fclose(fp2cs); fclose(fp2ci); fclose(fp2cj); }
		}
		if (writegm) fclose(fpgm); fclose(fpsig); if (ldebug) fclose(fpd); if (writegm) fclose(fpwq);
		if (mpiWorld->isHead() && elec->save_dHePhSum_disk) system("rm -rf ldbd_data/dHePhSum");
	}

	void compute_P(int ik, int jk, diagMatrix& Ek, diagMatrix& Ekp, FeynWann::StatePh& php, FeynWann::StatePh& phm, FeynWann::MatrixEph& mp, FeynWann::MatrixEph& mm,
		bool compute_imsig, bool ldebug, bool lindblad, bool lorentzian){
		ldebug = ldebug && lindblad && !lorentzian;
		// compute_imshig should only be true for one of compute_P in subroutine compute_eph
		double ethr = ePhDelta * nEphDelta;
		zeros(P1, (int)std::pow(nBandsSel, 4)); zeros(P2, (int)std::pow(nBandsSel, 4));
		//matrix s1 = elec->state_elec[ik].S[2](bStart, bStop, bStart, bStop), s2 = elec->state_elec[jk].S[2](bStart, bStop, bStart, bStop);
		matrix s1 = elec->state_elec[ik].S[2](bStart - elec->bBot_dm, bStop - elec->bBot_dm, bStart - elec->bBot_dm, bStop - elec->bBot_dm), 
			s2 = elec->state_elec[jk].S[2](bStart - elec->bBot_dm, bStop - elec->bBot_dm, bStart - elec->bBot_dm, bStop - elec->bBot_dm);
		matrix sdeg1 = degProj(s1, Ek, degthr), sdeg2 = degProj(s2, Ekp, degthr);

		for (int alpha = modeStart; alpha < modeStop; alpha++){
			if (alpha < modeSkipStop && alpha >= modeSkipStart) continue;
			double wqm = phm.omega[alpha], wqp = php.omega[alpha];
			if (wqm/Tmax <= omegabyTCut || wqp/Tmax <= omegabyTCut) continue;
			double sigma2 = std::pow(ePhDelta, 2);
			double prefac_sqrtexp, prefac_sqrtdelta, prefac_exp, prefac_delta, deltaplus, deltaminus;
			prefac_sqrtexp = -0.25 / sigma2; prefac_exp = -0.5 / sigma2;
			if (lorentzian)
				prefac_delta = ePhDelta / M_PI;
			else
				prefac_delta = 1. / (ePhDelta * sqrt(2.*M_PI));
			prefac_sqrtdelta = sqrt(prefac_delta);
			double prefac_imsig = M_PI / elec->nkTot * prefac_delta;

			double nqm = bose(std::max(1e-3, wqm / Tmax)), nqp = bose(std::max(1e-3, wqp / Tmax));
			if (ldebug) { fprintf(fpd, "alpha= %d wqm= %lg wqp= %lg nqm= %lg\n", alpha, wqm, wqp, nqm); fflush(fpd); }
			//matrix gp = dagger(mp.M[alpha](bStart, bStop, bStart, bStop)); // g^+_kk' = g^-_k'k^(dagger_n)
			//matrix gm = mm.M[alpha](bStart, bStop, bStart, bStop);
			matrix gp = dagger(mp.M[alpha]); // g^+_kk' = g^-_k'k^(dagger_n)
			matrix gm = mm.M[alpha];
			matrix sgpcomm = sdeg1 * gp - gp * sdeg2;
			matrix sgmcomm = sdeg1 * gm - gm * sdeg2;
			///*
			if (ldebug){
				//fprintf(fpd, "gp:");
				//for (int b1 = 0; b1 < nBandsSel; b1++)
				//for (int b2 = 0; b2 < nBandsSel; b2++)
				//	fprintf(fpd, " (%lg,%lg)", gp(b1, b2).real(), gp(b1, b2).imag());
				//fprintf(fpd, "\n:");
				fprintf(fpd, "gm:");  fflush(fpd);
				for (int b1 = 0; b1 < nBandsSel; b1++)
				for (int b2 = 0; b2 < nBandsSel; b2++){
					fprintf(fpd, " (%lg,%lg)", gm(b1, b2).real(), gm(b1, b2).imag()); fflush(fpd);
				}
				fprintf(fpd, "\n");  fflush(fpd);
			}
			//*/
			bool conserve = false;
			//G^+- = g^+- sqrt(delta(ek - ekp +- wq)); G^+_kk' = G^-_k'k^(dagger_n)
			for (int b1 = 0; b1 < nBandsSel; b1++)
			for (int b2 = 0; b2 < nBandsSel; b2++){
				bool inEwind = Ek[b1] >= Estart && Ek[b1] <= Estop && Ekp[b2] >= Estart && Ekp[b2] <= Estop;
				complex Gp = c0, G2p = c0, Gm = c0, G2m = c0;
				double dE = Ek[b1] - Ekp[b2];
				//double nq = !detailBalance ? nqp : bose(std::max(1e-3, -dE / Tmax));

				// phonon emission
				if (fabs(dE + wqp) < ethr && inEwind){
					conserve = true;
					if (lorentzian)
						deltaplus = 1. / (std::pow(dE + wqp, 2) + sigma2);
					else
						deltaplus = exp(prefac_exp*std::pow(dE + wqp, 2));
					if (lindblad)
						G2p = prefac_sqrtdelta * gp(b1, b2) * sqrt(deltaplus);
					else
						G2p = prefac_delta * gp(b1, b2) * deltaplus;

					if (compute_imsig && (ik != jk || b1 != b2)){
						//const vector3<>& v1 = elec->state_elec[ik].vVec[b1 + bStart]; const vector3<>& v2 = elec->state_elec[jk].vVec[b2 + bStart];
						const vector3<>& v1 = elec->state_elec[ik].vVec[b1]; const vector3<>& v2 = elec->state_elec[jk].vVec[b2];
						double cosThetaScatter = dot(v1, v2) / sqrt(std::max(1e-16, v1.length_squared() * v2.length_squared()));
						double dtmp1 = prefac_imsig * gp(b1, b2).norm() * (F[jk][b2] + nqp) * deltaplus;
						imsig[ik][b1] += dtmp1; imsigp[ik][b1] += dtmp1 * (1. - cosThetaScatter);
						imsigflip[ik][b1] += prefac_imsig * sgpcomm(b1, b2).norm() * (F[jk][b2] + nqp) * deltaplus;
						if (ik < jk){
							double dtmp2 = prefac_imsig * gp(b1, b2).norm() * (nqp + 1 - F[ik][b1]) * deltaplus;
							imsig[jk][b2] += dtmp2; imsigp[jk][b2] += dtmp2 * (1. - cosThetaScatter);
							imsigflip[jk][b2] += prefac_imsig * sgpcomm(b1, b2).norm() * (nqp + 1 - F[ik][b1]) * deltaplus;
						}
					}
				}
				Gp = lindblad ? G2p : gp(b1, b2);
				double dEbyT = dE / Tmax;
				double nq1_DB = (detailBalance && -dEbyT < 46) ? (exp(-dEbyT) * nqp) : nqp + 1; //when Ek + wqp = Ekp, nq + 1 = exp[(Ekp - Ek)/T] * nq
				App[b1*nBandsSel + b2] = lindblad ? Gp * sqrt(nq1_DB) : Gp * sqrt(nqp + 1);
				Apm[b1*nBandsSel + b2] = Gp * sqrt(nqp);
				A2pp[b1*nBandsSel + b2] = G2p * sqrt(nq1_DB);
				A2pm[b1*nBandsSel + b2] = G2p * sqrt(nqp);

				// phonon absorption
				//nq = !detailBalance ? nqm : bose(std::max(1e-3, dE / Tmax));
				if (fabs(dE - wqm) < ethr && inEwind){
					conserve = true;
					if (lorentzian)
						deltaminus = 1. / (std::pow(dE - wqm, 2) + sigma2);
					else
						deltaminus = exp(prefac_exp*std::pow(dE - wqm, 2));
					if (lindblad)
						G2m = prefac_sqrtdelta * gm(b1, b2) * sqrt(deltaminus);
					else
						G2m = prefac_delta * gm(b1, b2) * deltaminus;

					if (compute_imsig && (ik != jk || b1 != b2)){
						//const vector3<>& v1 = elec->state_elec[ik].vVec[b1 + bStart]; const vector3<>& v2 = elec->state_elec[jk].vVec[b2 + bStart];
						const vector3<>& v1 = elec->state_elec[ik].vVec[b1]; const vector3<>& v2 = elec->state_elec[jk].vVec[b2];
						double cosThetaScatter = dot(v1, v2) / sqrt(std::max(1e-16, v1.length_squared() * v2.length_squared()));
						double dtmp1 = prefac_imsig * gm(b1, b2).norm() * (nqm + 1 - F[jk][b2]) * deltaminus;
						imsig[ik][b1] += dtmp1; imsigp[ik][b1] += dtmp1 * (1. - cosThetaScatter);
						imsigflip[ik][b1] += prefac_imsig * sgmcomm(b1, b2).norm() * (nqm + 1 - F[jk][b2]) * deltaminus;
						if (ik < jk){
							double dtmp2 = prefac_imsig * gm(b1, b2).norm() * (F[ik][b1] + nqm) * deltaminus;
							imsig[jk][b2] += dtmp2; imsigp[jk][b2] += dtmp2 * (1. - cosThetaScatter);
							imsigflip[ik][b1] += prefac_imsig * sgmcomm(b1, b2).norm() * (F[ik][b1] + nqm) * deltaminus;
						}
					}
				}
				Gm = lindblad ? G2m : gm(b1, b2);
				nq1_DB = (detailBalance && dEbyT < 46) ? (exp(dEbyT) * nqm) : nqm + 1; //when Ekp + wqp = Ek, nq + 1 = exp[(Ek - Ekp)/T] * nq
				Amp[b1*nBandsSel + b2] = lindblad ? Gm * sqrt(nq1_DB) : Gm * sqrt(nqm + 1);
				Amm[b1*nBandsSel + b2] = Gm * sqrt(nqm);
				A2mp[b1*nBandsSel + b2] = G2m * sqrt(nq1_DB);
				A2mm[b1*nBandsSel + b2] = G2m * sqrt(nqm);
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
							P1[n12 + i3*nBandsSel + i4] += App[i13] * conj(A2pp[i2*nBandsSel + i4]) + Amm[i13] * conj(A2mm[i2*nBandsSel + i4]);
							P2[n12 + i3*nBandsSel + i4] += Amp[i31] * conj(A2mp[i4*nBandsSel + i2]) + Apm[i31] * conj(A2pm[i4*nBandsSel + i2]);
						}
					}
				}
			} // if (conserve)
		} // loop on alpha

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
		/*
		double prefac_imsig_fromP = M_PI / elec->nkTot;
		for (int i1 = 0; i1 < nBandsSel; i1++){
			int i11 = i1*nBandsSel + i1;
			int n11 = i11*nBandsSel*nBandsSel;
			for (int i2 = 0; i2 < nBandsSel; i2++){
				int i22 = i2*nBandsSel + i2;
				int n22 = i22*nBandsSel*nBandsSel;
				imsig[ik][i1] += prefac_imsig_fromP * (P1[n11 + i22].real() * F[jk][i2] + (1 - F[jk][i2]) * P2[n22 + i11].real());
				if (ik < jk)
					imsig[jk][i2] += prefac_imsig_fromP * (P2[n22 + i11].real() * F[ik][i1] + (1 - F[ik][i1]) * P1[n11 + i22].real());
			}
		}
		*/
	}
	void write_imsige(){
		string fnamesigkn = dir_ldbd + "ldbd_imsigkn.out";
		string fnamesigkn_bsq = dir_ldbd + "ldbd_imsigkn_bsq.out";
		string fnamesigkn_intp = dir_ldbd + "ldbd_imsigkn_intp.out";
		string fnamesigkn_flip = dir_ldbd + "ldbd_imsigkn_flip.out";
		FILE *fpsigkn = fopen(fnamesigkn.c_str(), "w");
		FILE *fpsigkn_bsq = fopen(fnamesigkn_bsq.c_str(), "w");
		FILE *fpsigkn_intp; if (fw.fwp.needLinewidth_ePh) fpsigkn_intp = fopen(fnamesigkn_intp.c_str(), "w");
		FILE *fpsigkn_flip = fopen(fnamesigkn_flip.c_str(), "w");
		fprintf(fpsigkn, "E(Har) ImSigma_kn(Har) ImSigmaP(Har) dfde v*v\n");
		fprintf(fpsigkn_bsq, "E(Har) ImSigma_kn(Har) b^2\n");
		if (fw.fwp.needLinewidth_ePh) fprintf(fpsigkn_intp, "E(Har) ImSigma_kn(Har) Interpolated ImSigma\n");
		fprintf(fpsigkn_flip, "E(Har) ImSigma_kn(Har) ImSigma_Flip\n");
		double imsig_max = imsig[0][0], imsig_min = imsig[0][0];
		for (size_t ik = 0; ik < k.size(); ik++){
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
			for (int b = 0; b < nBandsSel; b++){
				if (Ek[b] >= Estart && Ek[b] <= Estop){
					double dfde = F[ik][b] * (1 - F[ik][b]) / Tmax;
					//const vector3<>& v = elec->state_elec[ik].vVec[b + bStart];
					const vector3<>& v = elec->state_elec[ik].vVec[b];
					fprintf(fpsigkn, "%14.7le %14.7le %14.7le %14.7le %14.7le %14.7le %14.7le\n", Ek[b], imsig[ik][b], imsigp[ik][b], dfde, v[0] * v[0], v[1] * v[1], v[2] * v[2]);
					fprintf(fpsigkn_bsq, "%14.7le %14.7le %14.7le %14.7le %14.7le\n", Ek[b], imsig[ik][b], elec->bsq[ik][b][0], elec->bsq[ik][b][1], elec->bsq[ik][b][2]);
					//if (fw.fwp.needLinewidth_ePh) fprintf(fpsigkn_intp, "%14.7le %14.7le %14.7le\n", Ek[b], imsig[ik][b], elec->state_elec[ik].ImSigma_ePh(b + bStart, F[ik][b]));
					fprintf(fpsigkn_flip, "%14.7le %14.7le %14.7le\n", Ek[b], imsig[ik][b], imsigflip[ik][b]);
					if (imsig[ik][b] > imsig_max) imsig_max = imsig[ik][b];
					if (imsig[ik][b] < imsig_min) imsig_min = imsig[ik][b];
				}
			}
		}
		logPrintf("\nimsig_min = %lg eV imsig_max = %lg eV\n", imsig_min / eV, imsig_max / eV); logFlush();
		fclose(fpsigkn); fclose(fpsigkn_bsq); if (fw.fwp.needLinewidth_ePh){ fclose(fpsigkn_intp); } fclose(fpsigkn_flip);

		std::vector<double> imsige(102); std::vector<int> nstate(102);
		double Estart_imsige = Estart, Estop_imsige = Estop;
		if (!fw.isMetal){
			//if (ePhOnlyElec) Estart_imsige = minval(elec->state_elec, bCBM, bStop) - std::min(7., nEphDelta) * ePhDelta;
			//if (ePhOnlyHole) Estop_imsige = maxval(elec->state_elec, bStart, bCBM) + std::max(7., nEphDelta) * ePhDelta;
			if (ePhOnlyElec) Estart_imsige = minval(elec->state_elec, bCBM - elec->bRef, bStop - elec->bRef) - std::min(7., nEphDelta) * ePhDelta;
			if (ePhOnlyHole) Estop_imsige = maxval(elec->state_elec, bStart - elec->bRef, bCBM - elec->bRef) + std::max(7., nEphDelta) * ePhDelta;
			if (ePhOnlyElec || ePhOnlyHole) logPrintf("Active energy range for printing ImSigma(E): %.3lf to %.3lf eV\n", Estart_imsige / eV, Estop_imsige / eV);
		}
		double dE = (Estop_imsige - Estart_imsige) / 100;
		for (size_t ik = 0; ik < k.size(); ik++){
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
			for (int b = 0; b < nBandsSel; b++){
				int ie = round((Ek[b] - Estart_imsige) / dE);
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
				fprintf(fpsige, "%14.7le %14.7le %d\n", (Estart_imsige + ie*dE) / eV, imsige[ie] / eV, nstate[ie]);
			}
		}
		fclose(fpsige);
	}
	void merge_eph_P_mpi(){
		logPrintf("\n"); logFlush();
		// This subroutine requires that the index order is consistent with the file name order
		complex ctmp; int itmp;
		if (!write_sparseP){
			merge_files_mpi(dir_ldbd + "ldbd_P1_lindblad" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files_mpi(dir_ldbd + "ldbd_P2_lindblad" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4));
			if (needConventional) { merge_files_mpi(dir_ldbd + "ldbd_P1_conventional" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files_mpi(dir_ldbd + "ldbd_P2_conventional" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); }
		}
		else{
			merge_files_mpi(dir_ldbd + "sP1_lindblad" + shole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_lindblad" + shole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP1_lindblad" + shole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_lindblad" + shole + "_j.bin", itmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_lindblad" + shole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_lindblad" + shole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_lindblad" + shole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_lindblad" + shole + "_j.bin", itmp, 1);
			if (needConventional){
				merge_files_mpi(dir_ldbd + "sP1_conventional" + shole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_conventional" + shole + "_s.bin", ctmp, 1);
				merge_files_mpi(dir_ldbd + "sP1_conventional" + shole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_conventional" + shole + "_j.bin", itmp, 1);
				merge_files_mpi(dir_ldbd + "sP2_conventional" + shole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_conventional" + shole + "_s.bin", ctmp, 1);
				merge_files_mpi(dir_ldbd + "sP2_conventional" + shole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_conventional" + shole + "_j.bin", itmp, 1);
			}
		}
	}
	void merge_eph_P(){
		logPrintf("\n"); logFlush();
		// This subroutine requires that the index order is consistent with the file name order
		complex ctmp; int itmp;
		if (!write_sparseP){
			merge_files(dir_ldbd + "ldbd_P1_lindblad" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files(dir_ldbd + "ldbd_P2_lindblad" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4));
			if (needConventional){ merge_files(dir_ldbd + "ldbd_P1_conventional" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files(dir_ldbd + "ldbd_P2_conventional" + shole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); }
		}
		else{
			merge_files(dir_ldbd + "sP1_lindblad" + shole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP1_lindblad" + shole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP1_lindblad" + shole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP1_lindblad" + shole + "_j.bin", itmp, 1);
			merge_files(dir_ldbd + "sP2_lindblad" + shole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP2_lindblad" + shole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP2_lindblad" + shole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP2_lindblad" + shole + "_j.bin", itmp, 1);
			if (needConventional){
				merge_files(dir_ldbd + "sP1_conventional" + shole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP1_conventional" + shole + "_s.bin", ctmp, 1);
				merge_files(dir_ldbd + "sP1_conventional" + shole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP1_conventional" + shole + "_j.bin", itmp, 1);
				merge_files(dir_ldbd + "sP2_conventional" + shole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP2_conventional" + shole + "_s.bin", ctmp, 1);
				merge_files(dir_ldbd + "sP2_conventional" + shole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP2_conventional" + shole + "_j.bin", itmp, 1);
			}
		}
	}
	void merge_eph_gm(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (writegm && mpiWorld->isHead()){
			if (!keepgm){
				logPrintf("\nDelete gm:\n");
				for (int i = 0; i < mpiWorld->nProcesses(); i++){
					ostringstream convert; convert << i;
					string fnamegmi = dir_ldbd + "ldbd_gm.bin." + convert.str();
					remove(fnamegmi.c_str());
					string fnamewqi = dir_ldbd + "ldbd_wq_kpair.bin." + convert.str();
					remove(fnamewqi.c_str());
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	void savekpairData(){
		if (!read_kpairs){
			elec->saveSize(param, kpairs.size(), ph->omegaMax); logPrintf("saveSize done\n");
			saveKpair(); logPrintf("saveKpair done\n");
		}
		logPrintf("\nFor ePh: Estart= %lg eV Estop= %lg eV bStart= %d bStop= %d\n\n", Estart / eV, Estop / eV, bStart, bStop); logFlush();
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
		FILE *fp1, *fp2;
		FILE *fp1ns, *fp2ns, *fp1s, *fp2s, *fp1i, *fp2i, *fp1j, *fp2j;
		string scatt = (lindblad ? "lindblad" : "conventional") + shole;
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
		//double szeq = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart, bStop, elec->state_elec);
		double szeq = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart - elec->bBot_dm, bStop - elec->bBot_dm, elec->state_elec);
		for (size_t ik = 0; ik < k.size(); ik++){
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
			//matrix H1 = Bzpert * elec->state_elec[ik].S[2](bStart, bStop, bStart, bStop);
			matrix H1 = Bzpert * elec->state_elec[ik].S[2](bStart - elec->bBot_dm, bStop - elec->bBot_dm, bStart - elec->bBot_dm, bStop - elec->bBot_dm);
			matrix dRho = dRho_H1(Ek, F[ik], Tmax, H1, nBandsSel);
			for (int i = 0; i < nBandsSel; i++)
			for (int j = 0; j < nBandsSel; j++)
				dm[ik][i*nBandsSel + j] += dRho(i, j);
		}
		//double sz0 = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart, bStop, elec->state_elec);
		double sz0 = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart - elec->bBot_dm, bStop - elec->bBot_dm, elec->state_elec);
		set_dm1(dm, k.size(), nBandsSel, dm1);
		zeros(ddm, k.size(), nBandsSel*nBandsSel);

		P1_next = alloc_array((int)std::pow(nBandsSel, 4)); P2_next = alloc_array((int)std::pow(nBandsSel, 4));

		prefac_eph = 2 * M_PI / elec->nkTot;
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
		//double sz1 = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart, bStop, elec->state_elec);
		double sz1 = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart - elec->bBot_dm, bStop - elec->bBot_dm, elec->state_elec);
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

	std::vector<vector3<>> Gamma_ds;
	void T1_rate_dRho(bool drhosdeg, bool obsdeg, bool imsigma_eph, double fac_imsig, double impurity_broadening){ // impurity_broadening in eV
		ostringstream convert; convert << mpiWorld->iProcess();
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname1 and fname2 are not created for non-root processes
		string fnamegm = dir_ldbd + "ldbd_gm.bin." + convert.str();
		FILE *fpgm = fopen(fnamegm.c_str(), "rb");

		std::vector<std::vector<matrix>> Sfull(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		std::vector<std::vector<matrix>> Sdeg(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		std::vector<std::vector<matrix>> dRho(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
		for (size_t ik = 0; ik < k.size(); ik++){
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
			for (int id = 0; id < 3; id++){
				//Sfull[ik][id] = elec->state_elec[ik].S[id](bStart, bStop, bStart, bStop);
				Sfull[ik][id] = elec->state_elec[ik].S[id](bStart - elec->bBot_dm, bStop - elec->bBot_dm, bStart - elec->bBot_dm, bStop - elec->bBot_dm);
				degProj(Sfull[ik][id], Ek, degthr, Sdeg[ik][id]);
				if (drhosdeg)
					dRho[ik][id] = dRho_H1(Ek, F[ik], Tmax, Sdeg[ik][id], nBandsSel);
				else
					dRho[ik][id] = dRho_H1(Ek, F[ik], Tmax, Sfull[ik][id], nBandsSel);
			}
		}
		//vector3<> dS = compute_spin(dRho, k.size(), elec->nkTot, nBandsSel, bStart, bStop, elec->state_elec);
		vector3<> dS = compute_spin(dRho, k.size(), elec->nkTot, nBandsSel, bStart - elec->bBot_dm, bStop - elec->bBot_dm, elec->state_elec);

		vector3<> Sdot(0., 0., 0.);
		double prefac = 2 * M_PI / elec->nkTot / elec->nkTot;
		double ethr = nEphDelta * ePhDelta;
		if (drhosdeg && obsdeg) Gamma_ds.resize(ngrid, vector3<>(0,0,0));

		for (int ikpair_local = 0; ikpair_local < nkpairMine; ikpair_local++){
			int ikpair_glob = ikpair_local + ikpairStart;
			int ik0 = kpairs[ikpair_glob].first, jk0 = kpairs[ikpair_glob].second;

			// we need this trick since gm file stores g(jk,ik) right after g(ik,jk) if ik < jk but kpairs array stores only the pair (ik,jk) satisfying ik <= jk
			int nrun = (ik0 == jk0) ? 1 : 2;
			for (int irun = 0; irun < nrun; irun++){
				int ik, jk;
				if (irun == 0){ ik = ik0; jk = jk0; }
				else{ ik = jk0; jk = ik0; }

				//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop), Ekp = elec->state_elec[jk].E(bStart, bStop);
				diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef), Ekp = elec->state_elec[jk].E(bStart - elec->bRef, bStop - elec->bRef);
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
					if (im < modeSkipStop && im >= modeSkipStart) continue;
					double wq = ph.omega[im];
					double nq = bose(std::max(1e-3, wq / Tmax));
					if (!imsigma_eph){
						bIndex = 0;
						for (int b2 = 0; b2 < nBandsSel; b2++)
						for (int b1 = 0; b1 < nBandsSel; b1++){
							prefac_sqrtexp[bIndex] = -0.25 / std::pow(ePhDelta, 2);
							prefac_sqrtdelta[bIndex] = 1. / sqrt(ePhDelta * sqrt(2.*M_PI));
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
						if (fabs(dE - wq) < ethr){
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
					if (obsdeg){
						for (int id = 0; id < 3; id++){
							SGcomm[id] = Sdeg[ik][id] * G - G * Sdeg[jk][id];
							for (int b2 = 0; b2 < nBandsSel; b2++)
							for (int b1 = 0; b1 < nBandsSel; b1++){
								double dtmp = SGcomm[id](b1, b2).norm() * F[jk][b2] * (1 - F[ik][b1]) * nq / Tmax;
								contrib[id] += dtmp;
								double ds = abs(Sdeg[ik][id](b1, b1) - Sdeg[jk][id](b2, b2)) / 2.;
								int is = floor(ds * ngrid);
								if (is >= ngrid - 1) Gamma_ds[ngrid - 1][id] += dtmp;
								else Gamma_ds[is][id] += dtmp;
							}
						}
					}
					else{
						for (int id = 0; id < 3; id++)
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
				}
				Sdot += prefac * contrib;
			}
		}

		mpiWorld->allReduce(Sdot, MPIUtil::ReduceSum);
		if (drhosdeg && obsdeg){
			mpiWorld->allReduceData(Gamma_ds, MPIUtil::ReduceSum);
			string fnameds = dir_ldbd + "Gamma_ds.dat"; FILE *fpds = fopen(fnameds.c_str(), "w");
			std::vector<vector3<>> sumds(ngrid); sumds[0] = Gamma_ds[0];
			for (int is = 1; is < ngrid; is++)
				sumds[is] = sumds[is-1] + Gamma_ds[is];
			for (int is = 0; is < ngrid; is++)
				fprintf(fpds, "%lg %lg %lg %lg %lg %lg %lg\n", (is + 0.5) / ngrid,
					Gamma_ds[is][0] / sumds[ngrid - 1][0] * ngrid, Gamma_ds[is][1] / sumds[ngrid - 1][1] * ngrid, Gamma_ds[is][2] / sumds[ngrid - 1][2] * ngrid,
					sumds[is][0] / sumds[ngrid - 1][0], sumds[is][1] / sumds[ngrid - 1][1], sumds[is][2] / sumds[ngrid - 1][2]);
			fprintf(fpds, "T1 = %lg %lg %lg ps\n", -dS[0] / prefac / sumds[ngrid-1][0] / ps, -dS[1] / prefac / sumds[ngrid-1][1] / ps, -dS[2] / prefac / sumds[ngrid-1][2] / ps);
			fclose(fpds);
		}

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
		if (needConventional) T1_1step_useP(false, 0); // just to confirm zero dRho leads to zero change
		T1_1step_useP(true, 1 * Tesla);
		T1_1step_useP(true, 0.1 * Tesla);
		if (needConventional) T1_1step_useP(false, 0.1 * Tesla);
	}
	void relax_rate_usegm(){
		// single-rate calculations
		if (writegm){ // storing all gm matrices may need huge memory; we may not want to print it
			T1_rate_dRho(true, true, false, 0, 0);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with constant smearings:\n");
			logPrintf("**************************************************\n");
			if (nEphDelta * ePhDelta / eV / 0.001 > 4) T1_rate_dRho(false, false, true, 0, 0.001);
			if (nEphDelta * ePhDelta / eV / 0.005 > 4) T1_rate_dRho(false, false, true, 0, 0.005);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with ImSigma_eph + a constant:\n");
			logPrintf("**************************************************\n");
			T1_rate_dRho(false, false, true, 1, 0);
			T1_rate_dRho(false, false, true, 0.5, 0);
			if (nEphDelta * ePhDelta / eV / 0.001 > 4) T1_rate_dRho(false, false, true, 1, 0.001);
			if (nEphDelta * ePhDelta / eV / 0.005 > 4) T1_rate_dRho(false, false, true, 1, 0.005);
		}
	}

	// matrix element analysis
	void compute_eph_analyse_g2_E1E2fix();
};