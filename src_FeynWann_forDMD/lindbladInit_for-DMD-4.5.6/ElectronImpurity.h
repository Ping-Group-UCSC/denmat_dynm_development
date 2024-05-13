#pragma once
#include "common_headers.h"
#include "parameters.h"
#include "lattice.h"
#include "electron.h"
#include "phonon.h"
#include "sparse_matrix.h"
#include "ElectronPhonon.h"

//Lindblad initialization using FeynWann callback
class ElectronImpurity{
public:
	FeynWann& fw;
	parameters *param;
	lattice *latt;
	electron *elec;
	phonon *ph;
	ElectronPhonon *eph;

	double Tmax;

	bool read_kpairs;

	bool onlyInterValley, onlyIntraValley, eScattOnlyElec, eScattOnlyHole; //!< whether e-ph coupling is enabled
	const int iDefect; string siDefect, siDefectHole;
	const double defect_density; double defect_fraction;

	bool detailBalance; double omegaL;
	const double scattDelta; double nScattDelta; //!< Gaussian energy conservation width
	bool needConventional, writeg, keepg, mergeg, write_sparseP;

	double degthr;

	// ?Start and ?Stop are used most frequently and used as global variables in some subroutines
	// ?_probe for probe and can contain bands far from band edges; ?_eph for e-ph scattering; ?_dm for states related to density matrix change
	int bStart, bStop, bCBM, nBandsSel, nBandsSel_probe;
	double Estart, Estop; //energy range for k selection

	std::vector<vector3<>>& k; //selected k-points
	std::vector<double>& E; //all band energies for selected k-points
	std::vector<diagMatrix>& F;

	// matrix element analysis
	bool analyse_g2;

	ElectronImpurity(FeynWann& fw, lattice *latt, parameters *param, electron  *elec, phonon *ph)
		: fw(fw), latt(latt), param(param), elec(elec), ph(ph),
		Tmax(param->Tmax), read_kpairs(param->ePhEnabled ? true : param->read_kpairs),
		onlyInterValley(param->onlyInterValley), onlyIntraValley(param->onlyIntraValley), eScattOnlyElec(param->eScattOnlyElec), eScattOnlyHole(param->eScattOnlyHole),
		detailBalance(param->detailBalance_defect), omegaL(param->omegaL),
		scattDelta(param->scattDelta), nScattDelta(param->nScattDelta), degthr(param->degthr),
		needConventional(param->needConventional), writeg(param->writegm), keepg(param->keepgm), mergeg(param->mergegm), write_sparseP(param->write_sparseP),
		k(elec->k), E(elec->E), F(elec->F),
		bStart(elec->bStart), bStop(elec->bStop), bCBM(elec->bCBM), nBandsSel(elec->nBandsSel), nBandsSel_probe(elec->nBandsSel_probe),
		Estart(elec->Estart), Estop(elec->Estop),
		iDefect(param->iDefect), defect_density(param->defect_density * latt->cminvdim2au()), defect_fraction(defect_density*latt->cell_size()),
		analyse_g2(true)
	{
		shole = eScattOnlyHole ? "_hole" : "";
		ostringstream convert; convert << iDefect; convert.flush(); MPI_Barrier(MPI_COMM_WORLD);
		siDefect = "_D" + convert.str(); siDefectHole = siDefect + shole;
		logPrintf("\ndefect_fraction = %lg\n", defect_fraction);
	}

	//--------- k-pair selection -------------
	//std::vector<std::vector<size_t>> kpartners; //list of e-ph coupled k2 for each k1
	std::vector<std::pair<size_t, size_t>> kpairs; //pairs of k1 and k2
	inline void selectActive(const double*& Ebegin, const double*& Eend, double Elo, double Ehi){ //narrow pointer range to data within [Estart,Estop]
		Ebegin = std::lower_bound(Ebegin, Eend, Elo);
		Eend = &(*std::lower_bound(reverse(Eend), reverse(Ebegin), Ehi, std::greater<double>())) + 1;
	}
	inline void kpSelect(int ik1){
		//Find pairs of momentum conserving elastic scattering:
		for (size_t ik2 = ik1; ik2 < k.size(); ik2++){
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
				for (const double* E2 = E2begin; E2 < E2end; E2++){ //E2 in active range
					double dE = (*E1) - (*E2);
					if (fabs(dE) < nScattDelta*scattDelta)
						Econserve = true;
					if (Econserve) break;
				}
				if (Econserve) break;
			}
			if (Econserve) kpairs.push_back(std::make_pair(ik1, ik2));
		}
	}
	void kpairSelect(){
		logPrintf("\nFor kpairSelect: Estart= %lg Estop= %lg bStart= %d bStop= %d\n\n", Estart, Estop, bStart, bStop); logFlush();

		if (read_kpairs){
			assert(fileSize("ldbd_data/ldbd_size.dat") > 0);
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
			logPrintf("\nRead k paris from files.\n"); logFlush();
			assert(fileSize(fnamek.c_str()) > 0 && fileSize(fnamekp.c_str()) > 0);
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

		//Parallel:
		size_t kStart, kStop; //!< range of offstes handled by this process groups
		TaskDivision(k.size(), mpiWorld).myRange(kStart, kStop);
		size_t nkMine = kStop - kStart;
		size_t kInterval = std::max(1, int(round(nkMine / 50.))); //interval for reporting progress

		//Find momentum-conserving k-pairs for which energy conservation is also possible for some bands:
		logPrintf("Scanning k-pairs with e-ph coupling: "); logFlush();
		for (size_t ik1 = kStart; ik1 < kStop; ik1++){
			kpSelect(ik1);
			if ((ik1 - kStart + 1) % kInterval == 0) { logPrintf("%d%% ", int(round((ik1 - kStart + 1)*100. / nkMine))); logFlush(); }
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
	complex *P1, *P2, *A, *A2, *App, *Apm, *Amp, *Amm, *A2pp, *A2pm, *A2mp, *A2mm;
	double **imsig, **imsigp;
	
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
			fp1 = fopenP("ldbd_P1_lindblad" + siDefectHole + ".bin." + convert.str(), "wb"); fp2 = fopenP("ldbd_P2_lindblad" + siDefectHole + ".bin." + convert.str(), "wb");
			if (needConventional) { fp1c = fopenP("ldbd_P1_conventional" + siDefectHole + ".bin." + convert.str(), "wb"); fp2c = fopenP("ldbd_P2_conventional" + siDefectHole + ".bin." + convert.str(), "wb"); }
		}
		else{
			fp1ns = fopenP("sP1_lindblad" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp1s = fopenP("sP1_lindblad" + siDefectHole + "_s.bin." + convert.str(), "wb");
			fp1i = fopenP("sP1_lindblad" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp1j = fopenP("sP1_lindblad" + siDefectHole + "_j.bin." + convert.str(), "wb");
			fp2ns = fopenP("sP2_lindblad" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp2s = fopenP("sP2_lindblad" + siDefectHole + "_s.bin." + convert.str(), "wb");
			fp2i = fopenP("sP2_lindblad" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp2j = fopenP("sP2_lindblad" + siDefectHole + "_j.bin." + convert.str(), "wb");
			if (needConventional){
				fp1cns = fopenP("sP1_conventional" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp1cs = fopenP("sP1_conventional" + siDefectHole + "_s.bin." + convert.str(), "wb");
				fp1ci = fopenP("sP1_conventional" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp1cj = fopenP("sP1_conventional" + siDefectHole + "_j.bin." + convert.str(), "wb");
				fp2cns = fopenP("sP2_conventional" + siDefectHole + "_ns.bin." + convert.str(), "wb"); fp2cs = fopenP("sP2_conventional" + siDefectHole + "_s.bin." + convert.str(), "wb");
				fp2ci = fopenP("sP2_conventional" + siDefectHole + "_i.bin." + convert.str(), "wb"); fp2cj = fopenP("sP2_conventional" + siDefectHole + "_j.bin." + convert.str(), "wb");
			}
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
		App = alloc_array(nBandsSel*nBandsSel); Apm = alloc_array(nBandsSel*nBandsSel); Amp = alloc_array(nBandsSel*nBandsSel); Amm = alloc_array(nBandsSel*nBandsSel);
		A2pp = alloc_array(nBandsSel*nBandsSel); A2pm = alloc_array(nBandsSel*nBandsSel); A2mp = alloc_array(nBandsSel*nBandsSel); A2mm = alloc_array(nBandsSel*nBandsSel);
		P1 = alloc_array((int)std::pow(nBandsSel, 4)); P2 = alloc_array((int)std::pow(nBandsSel, 4));
		imsig = alloc_real_array(k.size(), nBandsSel); imsigp = alloc_real_array(k.size(), nBandsSel);

		// analyse matrix elements
		double sum_g2 = 0, sum_sg2 = 0, sum_weight = 0;
		double prefac_exp = -0.5 / std::pow(scattDelta, 2);
		std::vector<std::vector<matrix>> Sdeg;
		if (analyse_g2){
			Sdeg.resize(k.size(), std::vector<matrix>(3, matrix(nBandsSel, nBandsSel)));
			for (size_t ik = 0; ik < k.size(); ik++){
				//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
				diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
				for (int id = 0; id < 3; id++){
					//matrix Sfull = elec->state_elec[ik].S[id](bStart, bStop, bStart, bStop);
					matrix Sfull = elec->state_elec[ik].S[id](bStart - elec->bBot_dm, bStop - elec->bBot_dm, bStart - elec->bBot_dm, bStop - elec->bBot_dm);
					degProj(Sfull, Ek, degthr, Sdeg[ik][id]);
				}
			}
		}

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
			if (onlyInterValley && !latt->isInterValley(k[ik], k[jk])) continue;
			if (onlyIntraValley && latt->isInterValley(k[ik], k[jk])) continue;
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop), Ekp = elec->state_elec[jk].E(bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef), Ekp = elec->state_elec[jk].E(bStart - elec->bRef, bStop - elec->bRef);
			if (ldebug){
				for (int b = 0; b < nBandsSel; b++){
					fprintf(fpd, "Ek[%d]= %lg Ekp[%d]= %lg\n", b, Ek[b], b, Ekp[b]); fflush(fpd);
				}
			}
			FeynWann::MatrixDefect mD;
			fw.defectCalc(elec->state_elec[ik], elec->state_elec[jk], mD); // g^i_kk'

			if (writeg){
				//matrix g = mD.M(bStart, bStop, bStart, bStop);
				matrix g = mD.M;
				fwrite(g.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpg);
				if (ik < jk){
					//matrix gji = dagger(mD.M(bStart, bStop, bStart, bStop));
					matrix gji = dagger(mD.M);
					fwrite(gji.data(), 2 * sizeof(double), nBandsSel*nBandsSel, fpg);
				}
			}

			if (analyse_g2){
				int nrun = (ik == jk) ? 1 : 2;
				for (int irun = 0; irun < nrun; irun++){
					int ik1, ik2;
					if (irun == 0){ ik1 = ik; ik2 = jk; }
					else{ ik1 = jk; ik2 = ik; }

					//diagMatrix E1 = elec->state_elec[ik1].E(bStart, bStop), E2 = elec->state_elec[ik2].E(bStart, bStop);
					diagMatrix E1 = elec->state_elec[ik1].E(bStart - elec->bRef, bStop - elec->bRef), E2 = elec->state_elec[ik2].E(bStart - elec->bRef, bStop - elec->bRef);
					matrix S1z = Sdeg[ik1][2], S2z = Sdeg[ik2][2];
					//matrix g = (irun == 0) ? mD.M(bStart, bStop, bStart, bStop) : dagger(mD.M(bStart, bStop, bStart, bStop));
					matrix g = (irun == 0) ? mD.M(0, nBandsSel, 0, nBandsSel) : dagger(mD.M(0, nBandsSel, 0, nBandsSel));
					matrix SGcomm = S1z * g - g * S2z;

					for (int b1 = 0; b1 < nBandsSel; b1++)
					for (int b2 = 0; b2 < nBandsSel; b2++){
						double weight = F[ik2][b2] * (1 - F[ik1][b1]) * exp(prefac_exp * std::pow(E1[b1] - E2[b2], 2));

						sum_weight += weight;
						sum_g2 += weight * g(b1, b2).norm();
						sum_sg2 += weight * SGcomm(b1, b2).norm();
					}
				}
			}

			compute_P_eimp(ik, jk, Ek, Ekp, mD, true, ldebug, true, false); // gaussian smearing
			fwriteP(fp1, fp2, fp1ns, fp1s, fp1i, fp1j, fp2ns, fp2s, fp2i, fp2j);
			
			if (needConventional){
				compute_P_eimp(ik, jk, Ek, Ekp, mD, false, ldebug, false, false); // conventional, gaussian smearing
				fwriteP(fp1c, fp2c, fp1cns, fp1cs, fp1ci, fp1cj, fp2cns, fp2cs, fp2ci, fp2cj);
			}

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
			fclose(fp1); fclose(fp2); if (needConventional){ fclose(fp1c); fclose(fp2c); }
		}
		else{
			fclose(fp1ns); fclose(fp1s); fclose(fp1i); fclose(fp1j); fclose(fp2ns); fclose(fp2s); fclose(fp2i); fclose(fp2j);
			if (needConventional){ fclose(fp1cns); fclose(fp1cs); fclose(fp1ci); fclose(fp1cj); fclose(fp2cns); fclose(fp2cs); fclose(fp2ci); fclose(fp2cj); }
		}
		if (writeg) fclose(fpg); fclose(fpsig); if (ldebug) fclose(fpd); //if (writeg) fclose(fpwq);

		if (analyse_g2){
			mpiWorld->reduce(sum_weight, MPIUtil::ReduceSum);
			mpiWorld->reduce(sum_g2, MPIUtil::ReduceSum);
			mpiWorld->reduce(sum_sg2, MPIUtil::ReduceSum);

			logPrintf("Average |g|^2 = %lg\n", sum_g2 / sum_weight);
			logPrintf("Average |[s,g]|^2 = %lg\n", sum_sg2 / sum_weight);
			logPrintf("sum of weights = %lg (%lu)\n", sum_weight, (kpairs.size() * 2 - k.size())*nBandsSel*nBandsSel);
		}
	}

	void compute_P_eimp(int ik, int jk, diagMatrix& Ek, diagMatrix& Ekp, FeynWann::MatrixDefect& mD,
		bool compute_imsig, bool ldebug, bool lindblad, bool lorentzian){
		ldebug = ldebug && lindblad && !lorentzian;
		// compute_imshig should only be true for one of compute_P in subroutine compute_eph
		double ethr = scattDelta * nScattDelta;
		zeros(P1, (int)std::pow(nBandsSel, 4)); zeros(P2, (int)std::pow(nBandsSel, 4));

		double sigma2 = std::pow(scattDelta, 2);
		double prefac_sqrtexp, prefac_sqrtdelta, prefac_exp, prefac_delta, delta, deltaplus, deltaminus;
		prefac_sqrtexp = -0.25 / sigma2; prefac_exp = -0.5 / sigma2;
		if (lorentzian)
			prefac_delta = scattDelta / M_PI;
		else
			prefac_delta = 1. / (scattDelta * sqrt(2.*M_PI));
		if (detailBalance) prefac_delta *= 0.5;
		prefac_sqrtdelta = sqrt(prefac_delta);
		double prefac_imsig = M_PI / elec->nkTot * prefac_delta;

		//matrix g = mD.M(bStart, bStop, bStart, bStop);
		matrix g = mD.M;
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
					//const vector3<>& v1 = elec->state_elec[ik].vVec[b1 + bStart]; const vector3<>& v2 = elec->state_elec[jk].vVec[b2 + bStart];
					const vector3<>& v1 = elec->state_elec[ik].vVec[b1]; const vector3<>& v2 = elec->state_elec[jk].vVec[b2];
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

			if (detailBalance){
				conserve = false;
				complex Gp = c0, G2p = c0, Gm = c0, G2m = c0;
				double dE = Ek[b1] - Ekp[b2];
				// emission
				if (fabs(dE + omegaL) < ethr && inEwind){
					conserve = true;
					if (lorentzian)
						deltaplus = 1. / (std::pow(dE + omegaL, 2) + sigma2);
					else
						deltaplus = exp(prefac_exp*std::pow(dE + omegaL, 2));
					if (lindblad)
						G2p = prefac_sqrtdelta * g(b1, b2) * sqrt(deltaplus);
					else
						G2p = prefac_delta * g(b1, b2) * deltaplus;
				}
				Gp = lindblad ? G2p : g(b1, b2);
				double dEbyT = dE / Tmax;
				double facDB = (-dEbyT < 46) ? exp(-dEbyT / 2) : 1; //when Ek + wqp = Ekp, nq + 1 = exp[(Ekp - Ek)/T] * nq
				double facDB2 = 1;
				//if (typeDB == 2) facDB2 = 1 / facDB;
				//if (typeDB == 3) facDB2 = facDB;
				//if (typeDB == 4){ facDB2 = 1 / facDB; facDB = 1; }
				//if (typeDB == 5){ facDB2 = 0.5 + 0.5 / facDB; facDB = 0.5 * (facDB + 1); }
				App[b1*nBandsSel + b2] = Gp * facDB;
				Apm[b1*nBandsSel + b2] = Gp * facDB2;
				A2pp[b1*nBandsSel + b2] = G2p * facDB;
				A2pm[b1*nBandsSel + b2] = G2p * facDB2;

				// absorption
				if (fabs(dE - omegaL) < ethr && inEwind){
					conserve = true;
					if (lorentzian)
						deltaminus = 1. / (std::pow(dE - omegaL, 2) + sigma2);
					else
						deltaminus = exp(prefac_exp*std::pow(dE - omegaL, 2));
					if (lindblad)
						G2m = prefac_sqrtdelta * g(b1, b2) * sqrt(deltaminus);
					else
						G2m = prefac_delta * g(b1, b2) * deltaminus;
				}
				Gm = lindblad ? G2m : g(b1, b2);
				facDB = (dEbyT < 46) ? exp(dEbyT / 2) : 1; //when Ekp + wqp = Ek, nq + 1 = exp[(Ek - Ekp)/T] * nq
				facDB2 = 1;
				//if (typeDB == 2) facDB2 = 1 / facDB;
				//if (typeDB == 3) facDB2 = facDB;
				//if (typeDB == 4){ facDB2 = 1 / facDB; facDB = 1; }
				//if (typeDB == 5){ facDB2 = 0.5 + 0.5 / facDB; facDB = 0.5 * (facDB + 1); }
				Amp[b1*nBandsSel + b2] = Gm * facDB;
				Amm[b1*nBandsSel + b2] = Gm * facDB2;
				A2mp[b1*nBandsSel + b2] = G2m * facDB;
				A2mm[b1*nBandsSel + b2] = G2m * facDB2;
			}
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
						if (detailBalance){
							P1[n12 + i3*nBandsSel + i4] += App[i13] * conj(A2pp[i2*nBandsSel + i4]) + Amm[i13] * conj(A2mm[i2*nBandsSel + i4]);
							P2[n12 + i3*nBandsSel + i4] += Amp[i31] * conj(A2mp[i4*nBandsSel + i2]) + Apm[i31] * conj(A2pm[i4*nBandsSel + i2]);
						}
						else{
							P1[n12 + i3*nBandsSel + i4] += A[i13] * conj(A2[i2*nBandsSel + i4]);
							P2[n12 + i3*nBandsSel + i4] += A[i31] * conj(A2[i4*nBandsSel + i2]);
						}
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
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
			for (int b = 0; b < nBandsSel; b++){
				if (Ek[b] >= Estart && Ek[b] <= Estop){
					fprintf(fpsigkn, "%14.7le %14.7le %14.7le\n", Ek[b], imsig[ik][b], imsigp[ik][b]);
					if (imsig[ik][b] > imsig_max) imsig_max = imsig[ik][b];
					if (imsig[ik][b] < imsig_min) imsig_min = imsig[ik][b];
				}
			}
		}
		logPrintf("\nimsig_min = %lg eV imsig_max = %lg eV\n", imsig_min / eV, imsig_max / eV); logFlush();
		fclose(fpsigkn);

		std::vector<double> imsige(102); std::vector<int> nstate(102);
		double Estart_imsige = Estart, Estop_imsige = Estop;
		if (!fw.isMetal){
			//if (eScattOnlyElec) Estart_imsige = minval(elec->state_elec, bCBM, bStop) - std::min(7., nScattDelta) * scattDelta;
			//if (eScattOnlyHole) Estop_imsige = maxval(elec->state_elec, bStart, bCBM) + std::max(7., nScattDelta) * scattDelta;
			if (eScattOnlyElec) Estart_imsige = minval(elec->state_elec, bCBM - elec->bRef, bStop - elec->bRef) - std::min(7., nScattDelta) * scattDelta;
			if (eScattOnlyHole) Estop_imsige = maxval(elec->state_elec, bStart - elec->bRef, bCBM - elec->bRef) + std::max(7., nScattDelta) * scattDelta;
			if (eScattOnlyElec || eScattOnlyHole) logPrintf("Active energy range for printing ImSigma(E): %.3lf to %.3lf eV\n", Estart_imsige / eV, Estop_imsige / eV);
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
			if (needConventional) { merge_files_mpi(dir_ldbd + "ldbd_P1_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files_mpi(dir_ldbd + "ldbd_P2_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); }
		}
		else{
			merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			if (needConventional){
				merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_s.bin", ctmp, 1);
				merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP1_conventional" + siDefectHole + "_j.bin", itmp, 1);
				merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_s.bin", ctmp, 1);
				merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files_mpi(dir_ldbd + "sP2_conventional" + siDefectHole + "_j.bin", itmp, 1);
			}
		}
	}
	void merge_eimp_P(){
		// This subroutine requires that the index order is consistent with the file name order
		complex ctmp; int itmp;
		if (!write_sparseP){
			merge_files(dir_ldbd + "ldbd_P1_lindblad" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files(dir_ldbd + "ldbd_P2_lindblad" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4));
			if (needConventional){ merge_files(dir_ldbd + "ldbd_P1_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); merge_files(dir_ldbd + "ldbd_P2_conventional" + siDefectHole + ".bin", ctmp, (size_t)std::pow(nBandsSel, 4)); }
		}
		else{
			merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP1_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_s.bin", ctmp, 1);
			merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP2_lindblad" + siDefectHole + "_j.bin", itmp, 1);
			if (needConventional){
				merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_s.bin", ctmp, 1);
				merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP1_conventional" + siDefectHole + "_j.bin", itmp, 1);
				merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_ns.bin", itmp, 1); merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_s.bin", ctmp, 1);
				merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_i.bin", itmp, 1); merge_files(dir_ldbd + "sP2_conventional" + siDefectHole + "_j.bin", itmp, 1);
			}
		}
	}
	void merge_eimp_g(){
		MPI_Barrier(MPI_COMM_WORLD);
		if (writeg && mpiWorld->isHead()){
			if (!keepg){
				logPrintf("Delete g:\n");
				for (int i = 0; i < mpiWorld->nProcesses(); i++){
					ostringstream convert; convert << i;
					string fnamegi = dir_ldbd + "ldbd_g.bin." + convert.str();
					remove(fnamegi.c_str());
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
		logPrintf("\nFor E-I: Estart= %lg eV Estop= %lg eV bStart= %d bStop= %d\n\n", Estart / eV, Estop / eV, bStart, bStop); logFlush();
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
		//double szeq = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart, bStop, elec->state_elec);
		double szeq = compute_sz(dm, k.size(), elec->nkTot, nBandsSel, bStart - elec->bBot_dm, bStop - elec->bBot_dm, elec->state_elec);
		for (size_t ik = 0; ik < k.size(); ik++){
			//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop);
			//matrix H1 = Bzpert * elec->state_elec[ik].S[2](bStart, bStop, bStart, bStop);
			diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef);
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

		prefac_scatt = 2 * M_PI / elec->nkTot;
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
		double prefac = 2 * M_PI / elec->nkTot / elec->nkTot * defect_fraction;
		double prefacFGR = 0.5 * prefac / Tmax;
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

				//diagMatrix Ek = elec->state_elec[ik].E(bStart, bStop), Ekp = elec->state_elec[jk].E(bStart, bStop);
				diagMatrix Ek = elec->state_elec[ik].E(bStart - elec->bRef, bStop - elec->bRef), Ekp = elec->state_elec[jk].E(bStart - elec->bRef, bStop - elec->bRef);

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
				if (obsdeg){
					for (int id = 0; id < 3; id++){
						SGcomm[id] = Sdeg[ik][id] * G - G * Sdeg[jk][id];
						for (int b2 = 0; b2 < nBandsSel; b2++)
						for (int b1 = 0; b1 < nBandsSel; b1++)
							contrib[id] += SGcomm[id](b1, b2).norm() * F[jk][b2] * (1 - F[ik][b1]);
					}
					Sdot += prefacFGR * contrib;
				}
				else{
					for (int id = 0; id < 3; id++)
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
		if (needConventional) T1_1step_useP(false, 0); // just to confirm zero dRho leads to zero change
		T1_1step_useP(true, 1 * Tesla);
		T1_1step_useP(true, 0.1 * Tesla);
		if (needConventional) T1_1step_useP(false, 0.1 * Tesla);
	}
	void relax_rate_useg(){
		// single-rate calculations
		if (writeg){ // storing all g matrices may need huge memory; we may not want to print it
			T1_rate_dRho(true, true, false, 0, 0);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with constant smearings:\n");
			logPrintf("**************************************************\n");
			if (nScattDelta * scattDelta / eV / 0.001 > 4) T1_rate_dRho(false, false, true, 0, 0.001);
			if (nScattDelta * scattDelta / eV / 0.005 > 4) T1_rate_dRho(false, false, true, 0, 0.005);

			logPrintf("\n**************************************************\n");
			logPrintf("dRho formula with ImSigma_eph + a constant:\n");
			logPrintf("**************************************************\n");
			T1_rate_dRho(false, false, true, 1, 0);
			T1_rate_dRho(false, false, true, 0.5, 0);
			if (nScattDelta * scattDelta / eV / 0.001 > 4) T1_rate_dRho(false, false, true, 1, 0.001);
			if (nScattDelta * scattDelta / eV / 0.005 > 4) T1_rate_dRho(false, false, true, 1, 0.005);
		}
	}
};