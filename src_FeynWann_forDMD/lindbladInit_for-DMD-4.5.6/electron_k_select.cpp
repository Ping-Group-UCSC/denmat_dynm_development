#include "electron.h"

void electron::get_k_offsets(vector3<int> NkMult){
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

	nkTot = (double)NkFine[0] * (double)NkFine[1] * (double)NkFine[2];
}
void electron::read_gfac(){
	if (read_gfack){
		// read_gfack requires g_tensor_k.dat exists
		logPrintf("read g_tensor_k.dat\n"); logFlush();
		assert(fileSize("g_tensor_k.dat") > 0);
		FILE *fp = fopen("g_tensor_k.dat", "r");
		char s[200]; fgets(s, sizeof s, fp);
		while (fgets(s, sizeof s, fp) != NULL){
			if (s[0] != '\n'){
				matrix3<> m3tmp;
				sscanf(s, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
					&m3tmp(0, 0), &m3tmp(0, 1), &m3tmp(0, 2),
					&m3tmp(1, 0), &m3tmp(1, 1), &m3tmp(1, 2),
					&m3tmp(2, 0), &m3tmp(2, 1), &m3tmp(2, 2));
				gfack.push_back(m3tmp);
			}
		}
		fclose(fp);
		logPrintf("size of gfack: %lu\n", gfack.size()); logFlush();
		matrix3<> mean_gfack = mean_of_(gfack);
		matrix3<> sigma_gfack = sigma_of_(gfack);
		logPrintf("mean of gfack without weights:\n"); logFlush();
		if (mpiWorld->isHead()) mean_gfack.print(stdout, " %lg"); logFlush();
		logPrintf("sigma of gfack without weights:\n"); logFlush();
		if (mpiWorld->isHead()) sigma_gfack.print(stdout, " %lg"); logFlush();
	}
}
void electron::read_Bso(){
	if (read_Bsok){
		// read_Bsok requires Bso_vector_k.dat exists
		logPrintf("read Bso_vector_k.dat\n"); logFlush();
		assert(fileSize("Bso_vector_k.dat") > 0);
		FILE *fp = fopen("Bso_vector_k.dat", "r");
		char s[200]; fgets(s, sizeof s, fp);
		if (Bsok.size() == 0){
			while (fgets(s, sizeof s, fp) != NULL){
				if (s[0] != '\n'){
					vector3<> v3tmp;
					sscanf(s, "%lf %lf %lf", &v3tmp[0], &v3tmp[1], &v3tmp[2]);
					Bsok.push_back(v3tmp);
				}
			}
			logPrintf("size of Bsok: %lu\n", Bsok.size()); logFlush();
		}
		else{
			int ik = 0;
			while (fgets(s, sizeof s, fp) != NULL){
				if (s[0] != '\n'){
					vector3<> v3tmp;
					sscanf(s, "%lf %lf %lf", &v3tmp[0], &v3tmp[1], &v3tmp[2]);
					Bsok[ik] += v3tmp;
					ik++;
				}
			}
			logPrintf("number of Bsok read from file: %lu\n", ik); logFlush();
		}
		fclose(fp);
	}
}

//--------- number of DFT bands skipped in Wannier -------------
void electron::get_band_skipped(){
	if (band_skipped < 0) return;
	assert(band_skipped + fw.nBands <= fw.nBandsDFT);
	logPrintf("\nfind number of DFT bands skipped in Wannier\n");
	//Read DFT energies at Gamma
	diagMatrix Edft(fw.nBandsDFT);
	string fname = fw.fwp.totalEprefix + ".eigenvals";
	assert(fileSize(fname.c_str()) > 0);
	FILE *fp = fopen(fname.c_str(), "rb");
	fread(Edft.data(), sizeof(double), fw.nBandsDFT, fp); fclose(fp);
	axbyc(Edft.data(), nullptr, fw.nBandsDFT, 0, 1, -fw.mu); // Edft = Edft - mu
	fw.fwp.applyScissor(Edft);

	vector3<> Bext(fw.fwp.Bext); double EzExt = fw.fwp.EzExt;
	fw.fwp.Bext[0] = fw.fwp.Bext[1] = fw.fwp.Bext[2] = 0; fw.fwp.EzExt = 0; // temporarily set external field as zero
	FeynWann::StateE e;
	fw.eCalc(vector3<>(), e); // Wannier state
	fw.fwp.Bext[0] = Bext[0]; fw.fwp.Bext[1] = Bext[1]; fw.fwp.Bext[2] = Bext[2]; fw.fwp.EzExt = EzExt; // set external field back
	std::vector<double> w(fw.nBands);
	for (int b = 0; b < fw.nBands; b++)
		w[b] = exp(-std::pow(e.E[b], 2)/2./std::pow(0.001,2));

	diagMatrix de(fw.nBands);
	axbyc(&de[0], &e.E[0], fw.nBands); // de = e.E
	axbyc(&de[0], &Edft[band_skipped], fw.nBands, -1, 1); // de = de - Edft[band_skipped:band_skipped+fw.nBands]
	double err_min = sigma_of_array(&de[0], fw.nBands, false, 0, w.data()); // compute sqrt(sum_b de[b]^2 * w[b] / sum_b w[b])

	int band_skipped_new = band_skipped;
	for (int i = 0; i < std::min(nValence, fw.nBandsDFT - fw.nBands); i++){
		if (i != band_skipped){
			assert(band_skipped + fw.nBands <= fw.nBandsDFT);
			axbyc(&de[0], &e.E[0], fw.nBands); // de = e.E
			axbyc(&de[0], &Edft[i], fw.nBands, -1, 1); // de = de - Edft[b:b+fw.nBands]
			double err = sigma_of_array(&de[0], fw.nBands, false, 0, w.data()); // compute sqrt(sum_b de[b]^2 * w[b] / sum_b w[b])
			if (err < err_min){
				band_skipped_new = i;
				err_min = err;
			}
		}
	}
	band_skipped = band_skipped_new;
	logPrintf("band_skipped = %d\n", band_skipped); logFlush();
}
//--------- energy selection -------------
inline void electron::eRange(const FeynWann::StateE& state){
	if (band_skipped < 0){
		for (const double& E : state.E){
			if (E < 1e-4 and E > EvMax){
				EvMax = E;
				k_EvMax = state.k;
			}
			if (E > 1e-4 and E < EcMin){
				EcMin = E;
				k_EcMin = state.k;
			}
		}
	}
	else{
		if (state.E[nValence - band_skipped - 1] > EvMax){
			EvMax = state.E[nValence - band_skipped - 1];
			k_EvMax = state.k;
		}
		if (state.E[nValence - band_skipped] < EcMin){
			EcMin = state.E[nValence - band_skipped];
			k_EcMin = state.k;
		}
	}
}
void electron::eRange(const FeynWann::StateE& state, void* params){
	((electron*)params)->eRange(state);
}

//--------- band selection -------------
inline void electron::bSelect(const FeynWann::StateE& state){
	for (int b = 0; b < fw.nBands; b++){
		const double& E = state.E[b];
		if (E >= Estart and b < bStart) bStart = b;
		if (E <= Estop and b >= bStop) bStop = b + 1;
		if (E <= Emid and b >= bCBM) bCBM = b + 1;
	}
}
void electron::bSelect(const FeynWann::StateE& state, void* params){
	((electron*)params)->bSelect(state);
}
void electron::bSelect_driver(const std::vector<vector3<>>& k0, const double& EBot, const double& ETop, int& bBot, int& bTop){
	Estart = EBot; Estop = ETop;
	bStart = fw.nBands; bStop = 0; bCBM = 0;
	if (!useFinek_for_ERange){
		//fw.eLoop(vector3<>(), electron::bSelect, this);
		for (vector3<> qOff : fw.qOffset)
			fw.eLoop(qOff, electron::bSelect, this);
	}
	else{
		//Parallel:
		size_t oStart, oStop; //range of offstes handled by this process group
		if (mpiGroup->isHead()) TaskDivision(k0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart); mpiGroup->bcast(oStop);
		size_t noMine = oStop - oStart;
		size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

		logPrintf("\nDetermin band range using fine k meshes\n"); logFlush();
		for (size_t o = oStart; o < oStop; o++){
			for (vector3<> qOff : fw.qOffset) fw.eLoop(k0[o] + qOff, electron::bSelect, this);
			if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); } //Print progress:
		}
		logPrintf("done.\n"); logFlush();
	}
	mpiWorld->allReduce(bStart, MPIUtil::ReduceMin);
	mpiWorld->allReduce(bStop, MPIUtil::ReduceMax);
	mpiWorld->allReduce(bCBM, MPIUtil::ReduceMax);
	bBot = bStart; bTop = bStop;
}

//--------- k-point selection -------------
inline void electron::kSelect(const FeynWann::StateE& state){
	bool active = false;
	//bool has_elec = false, has_hole = false;
	for (double E : state.E)
	if (E >= Estart and E <= Estop){
		if (!analyse_g2_E1E2fix) { active = true; break; }
		else if (fabs(E - E1fix) < thr_delta || fabs(E - E2fix) < thr_delta){ active = true; break; }
	}
	if (active){
		k.push_back(state.k);
		E.insert(E.end(), &state.E[bBot_probe], &state.E[bTop_probe]);
	}
}
void electron::kSelect(const FeynWann::StateE& state, void* params){
	((electron*)params)->kSelect(state);
}
void electron::kpointSelect(const std::vector<vector3<>>& k0){
	//Parallel:
	size_t oStart, oStop; //range of offstes handled by this process group
	if (mpiGroup->isHead()) TaskDivision(k0.size(), mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart); mpiGroup->bcast(oStop);
	size_t noMine = oStop - oStart;
	size_t oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

	// Determine number of DFT bands skipped in Wannier
	get_band_skipped();
	// Determine VBM and CBM
	EvMax = -DBL_MAX; EcMin = +DBL_MAX;
	if (read_erange_brange){
		logPrintf("\nRead energy range from ldbd_data/ldbd_erange_brange.dat\n"); logFlush();
		assert(exists("ldbd_data/ldbd_erange_brange.dat"));
		FILE *fp = fopen("ldbd_data/ldbd_erange_brange.dat", "r");
		char s[200];
		if (fgets(s, sizeof s, fp) != NULL)
			sscanf(s, "%le %le", &EvMax, &EcMin);
		fclose(fp);
	}
	else{
		//fw.eLoop(vector3<>(), electron::eRange, this);
		if (!useFinek_for_ERange){
			logPrintf("\nDetermin VBM and CBM using DFT k meshes\n"); logFlush();
			string fname = fw.fwp.totalEprefix + ".eigenvals";
			bool hasField = fw.fwp.EzExt or fw.fwp.Bext.length_squared() or read_Bsok or (Bin_model != "none" and alpha_Bin != 0);
			if (exists(fname.c_str()) and !hasField){
				logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
				ManagedArray<double> Edft; Edft.init(fw.nBandsDFT*fw.nStatesDFT);
				Edft.read(fname.c_str());
				axbyc(Edft.data(), nullptr, fw.nBandsDFT*fw.nStatesDFT, 0, 1, -fw.mu); // Edft = Edft - mu
				if (fw.fwp.scissor > 0){ for (double& E : Edft) { if (E > fw.fwp.degeneracyThreshold) E += fw.fwp.scissor; } }
				for (int k = 0; k < fw.nStatesDFT; k++){
					EvMax = std::max(EvMax, Edft.data()[k*fw.nBandsDFT + nValence - 1]); //highest valence eigenvalue at each q
					EcMin = std::min(EcMin, Edft.data()[k*fw.nBandsDFT + nValence]); //highest valence eigenvalue at each q
				}
			}
			else{
				for (vector3<> qOff : fw.qOffset) fw.eLoop(qOff, electron::eRange, this);
			}
			logPrintf("done.\n");
		}
		else{
			logPrintf("\nDetermin VBM and CBM using fine k meshes\n"); logFlush();
			for (size_t o = oStart; o < oStop; o++){
				for (vector3<> qOff : fw.qOffset) fw.eLoop(k0[o] + qOff, electron::eRange, this);
				if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); } //Print progress:
			}
			logPrintf("done.\n"); logFlush();
			//find out k of CBM and VBM
			double vbmArr[mpiWorld->nProcesses()], cbmArr[mpiWorld->nProcesses()];
			vector3<> kvbmArr[mpiWorld->nProcesses()], kcbmArr[mpiWorld->nProcesses()];
			for (int i = 0; i < mpiWorld->nProcesses(); i++){
				if (i == mpiWorld->iProcess()){
					vbmArr[i] = EvMax; kvbmArr[i] = k_EvMax;
					cbmArr[i] = EcMin; kcbmArr[i] = k_EcMin;
				}
				mpiWorld->bcast(vbmArr[i], i); mpiWorld->bcast(kvbmArr[i], i);
				mpiWorld->bcast(cbmArr[i], i); mpiWorld->bcast(kcbmArr[i], i);
			}
			double vbm = vbmArr[0], cbm = cbmArr[0];
			k_EvMax = kvbmArr[0], k_EcMin = kcbmArr[0];
			for (int i = 1; i < mpiWorld->nProcesses(); i++){
				if (vbmArr[i] > vbm){
					vbm = vbmArr[i];
					k_EvMax = kvbmArr[i];
				}
				if (cbmArr[i] < cbm){
					cbm = cbmArr[i];
					k_EcMin = kcbmArr[i];
				}
			}
			//logPrintf("VBM: %.6lf eV, CBM: %.6lf eV\n", vbm / eV, cbm / eV);
			logPrintf("VBM at %lg %lg %lg, CBM at %lg %lg %lg\n", k_EvMax[0], k_EvMax[1], k_EvMax[2], k_EcMin[0], k_EcMin[1], k_EcMin[2]);
			mpiWorld->allReduce(EvMax, MPIUtil::ReduceMax); mpiWorld->allReduce(EcMin, MPIUtil::ReduceMin);
		}
	}
	Emid = (EvMax + EcMin) / 2.;
	logPrintf("VBM: %.6lf eV, CBM: %.6lf eV, Middle: %.6lf eV\n", EvMax / eV, EcMin / eV, Emid / eV);
	logPrintf("Note that VBM and CBM may not be determined correctly,\nyou may have to use band_skipped to set starting band index of wannier bands\n");
	if (EvMax < EcMin && band_skipped >= 0) fw.isMetal = false;
	if (assumeMetal) fw.isMetal = true;
	// for debug
	if (!fw.isMetal){
		if (E1fix >= 0) E1fix = EcMin + E1fix;
		else E1fix = EvMax + E1fix;
		if (E2fix >= 0) E2fix = EcMin + E2fix;
		else E2fix = EvMax + E2fix;
	}
	if (analyse_g2_E1E2fix){
		logPrintf("E1fix = %.6lf eV E2fix = %.6lf eV thr_delta = %.3lf meV\n", E1fix / eV, E2fix / eV, thr_delta / eV * 1000); logFlush();
	}

	//Determine energy range:
	//--- add margins of max phonon energy, energy conservation width and fermiPrime width
	double Emargin = nkBT*Tmax; //neglect below 10^-3 occupation deviation from equilibrium
	if (ETop_set > EBot_set){
		EBot_dm = EBot_set;
		EBot_eph = (!fw.isMetal && eScattOnlyElec) ? Emid : EBot_dm;
		ETop_dm = ETop_set;
		ETop_eph = (!fw.isMetal && eScattOnlyHole) ? Emid : ETop_dm;
	}
	else if (!fw.isMetal){
		logPrintf("\nThe system is not metalic\n");
		double EBot_pump = EcMin - pumpOmegaMax - 5. / pumpTau;
		EBot_eph = eScattOnlyElec ? Emid : std::min(EBot_pump, std::min(EvMax, dmuMin) - Emargin);
		EBot_dm = std::min(EBot_pump, EBot_eph);

		double ETop_pump = EvMax + pumpOmegaMax + 5. / pumpTau;
		ETop_eph = eScattOnlyHole ? Emid : std::max(ETop_pump, std::max(EcMin, dmuMax) + Emargin);
		ETop_dm = std::max(ETop_pump, ETop_eph);
	}
	else{
		logPrintf("\nThe system is metalic\n");
		EBot_eph = dmuMin - pumpOmegaMax - std::max(5. / pumpTau + 3.*Tmax, Emargin);
		EBot_dm = EBot_eph;
		ETop_eph = dmuMax + pumpOmegaMax + std::max(5. / pumpTau + 3.*Tmax, Emargin);
		ETop_dm = ETop_eph;
	}
	EBot_probe = EBot_dm - probeOmegaMax - 5. / pumpTau;
	ETop_probe = ETop_dm + probeOmegaMax + 5. / pumpTau;
	logPrintf("Emargin = %.3lf eV\n", Emargin / eV);
	logPrintf("Active energy range for probe: %.3lf to %.3lf eV\n", EBot_probe / eV, ETop_probe / eV);
	logPrintf("Active energy range for kpointSelect: %.3lf to %.3lf eV\n", EBot_eph / eV, ETop_eph / eV);
	logPrintf("Active energy range for density matrix: %.3lf to %.3lf eV\n", EBot_dm / eV, ETop_dm / eV);

	//Select bands:
	if (read_erange_brange){
		logPrintf("\nRead energy range from ldbd_data/ldbd_erange_brange.dat\n"); logFlush();
		assert(exists("ldbd_data/ldbd_erange_brange.dat"));
		FILE *fp = fopen("ldbd_data/ldbd_erange_brange.dat", "r");
		char s[200]; fgets(s, sizeof s, fp);
		if (fgets(s, sizeof s, fp) != NULL)
			sscanf(s, "%d %d %d %d %d %d %d", &bBot_probe, &bTop_probe, &bBot_dm, &bTop_dm, &bBot_eph, &bTop_eph, &bCBM);
		fclose(fp);
	}
	else{
		bSelect_driver(k0, EBot_probe, ETop_probe, bBot_probe, bTop_probe);
		bSelect_driver(k0, EBot_dm, ETop_dm, bBot_dm, bTop_dm);
		bSelect_driver(k0, EBot_eph, ETop_eph, bBot_eph, bTop_eph);
	}
	nBandsSel_probe = bTop_probe - bBot_probe; bRef = bBot_probe;
	if (fw.isMetal) bCBM = bBot_probe;
	logPrintf("\nbBot_probe= %d bTop_probe= %d\n", bBot_probe, bTop_probe);
	logPrintf("bBot_dm= %d bTop_dm= %d\n", bBot_dm, bTop_dm);
	logPrintf("bBot_eph= %d bTop_eph= %d bCBM= %d\n", bBot_eph, bTop_eph, bCBM);

	//write energy range and band range
	if (mpiWorld->isHead()){
		FILE *fp = fopen("ldbd_data/ldbd_erange_brange.dat", "w");
		fprintf(fp, "%21.14le %21.14le\n", EvMax, EcMin);
		fprintf(fp, "%d %d %d %d %d %d %d\n", bBot_probe, bTop_probe, bBot_dm, bTop_dm, bBot_eph, bTop_eph, bCBM);
		fclose(fp);
	}

	//Select k-points:
	Estart = EBot_eph; Estop = ETop_eph;
	bStart = bBot_eph; bStop = bTop_eph; nBandsSel = bStop - bStart;

	//read k points from existing files
	if (read_kpts){
		if (fw.isMetal || !assumeMetal_scatt){
			logPrintf("\nRead k points from ldbd_data/ldbd_kvec.bin.\n"); logFlush();
			assert(exists("ldbd_data/ldbd_size.dat"));
			FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
			char s[200]; fgets(s, sizeof s, fp); fgets(s, sizeof s, fp);
			if (fgets(s, sizeof s, fp) != NULL){
				double dtmp; int nk; sscanf(s, "%le %d", &dtmp, &nk);
				k.resize(nk, vector3<>(0, 0, 0)); E.resize(nk*nBandsSel_probe); logPrintf("number of k = %lu\n", k.size());
			}
			fclose(fp);

			assert(exists("ldbd_data/ldbd_kvec.bin")); assert(exists("ldbd_data/ldbd_ek.bin"));
			FILE *fpk = fopen("ldbd_data/ldbd_kvec.bin", "rb"), *fpe = fopen("ldbd_data/ldbd_ek.bin", "rb");
			size_t expected_size_k = k.size() * 3 * sizeof(double), expected_size_e = k.size() * nBandsSel_probe * sizeof(double);
			check_file_size(fpk, expected_size_k, "ldbd_kvec.bin size does not match expected size"); check_file_size(fpe, expected_size_e, "ldbd_ek.bin size does not match expected size");
			fread(k.data(), sizeof(double), k.size() * 3, fpk);
			fread(E.data(), sizeof(double), k.size() * nBandsSel_probe, fpe);
			fclose(fpk); fclose(fpe);

			if (gfack.size() > 0) assert(gfack.size() == k.size());
			if (Bsok.size() > 0) assert(Bsok.size() == k.size());
		}
		else{
			logPrintf("\nRead k points from ldbd_data/ldbd_kvec_morek.bin.\n"); logFlush();
			assert(exists("ldbd_data/ldbd_kvec_morek.bin")); assert(exists("ldbd_data/ldbd_ek_morek.bin"));
			int nk = fileSize("ldbd_data/ldbd_kvec_morek.bin") / (3 * sizeof(double));
			k.resize(nk, vector3<>(0, 0, 0)); E.resize(nk*nBandsSel_probe); logPrintf("number of k = %lu\n", k.size());

			MPI_Barrier(MPI_COMM_WORLD);
			FILE *fpk = fopen("ldbd_data/ldbd_kvec_morek.bin", "rb"), *fpe = fopen("ldbd_data/ldbd_ek_morek.bin", "rb");
			size_t expected_size_k = k.size() * 3 * sizeof(double), expected_size_e = k.size() * nBandsSel_probe * sizeof(double);
			check_file_size(fpk, expected_size_k, "ldbd_kvec_morek.bin size does not match expected size"); check_file_size(fpe, expected_size_e, "ldbd_ek_morek.bin size does not match expected size");
			fread(k.data(), sizeof(double), k.size() * 3, fpk);
			fread(E.data(), sizeof(double), k.size() * nBandsSel_probe, fpe);
			fclose(fpk); fclose(fpe);
		}
		return;
	}

	//core part: selecting k points having energies without the energy window
	double otot = 0, oskip = 0; //for select_k_use_meff
	double ethr = std::max(EvMax - Estart, Estop - EcMin); //for select_k_use_meff
	double k2thr = 1.5 * (2 * meff*ethr); //1.5 to ensure enough k points are included
	matrix3<> offsetDimInv = inv(Diag(vector3<>(fw.offsetDim)));
	if (select_k_use_meff) logPrintf("ethr = %lg  k2thr = %lg  offsetDimInv = %lg %lg %lg\n", ethr, k2thr, offsetDimInv(0, 0), offsetDimInv(1, 1), offsetDimInv(2, 2));

	logPrintf("Scanning k-points with active states: "); logFlush();
	for (size_t o = oStart; o<oStop; o++){
		//fw.eLoop(k0[o], electron::kSelect, this);
		for (vector3<> qOff : fw.qOffset){
			if (select_k_use_meff){
				vector3<int> iOff;
				bool nearValley = false;
				for (iOff[0] = 0; iOff[0] < fw.offsetDim[0]; iOff[0]++){
					for (iOff[1] = 0; iOff[1] < fw.offsetDim[1]; iOff[1]++){
						for (iOff[2] = 0; iOff[2] < fw.offsetDim[2]; iOff[2]++){
							vector3<> k = offsetDimInv * iOff + k0[o] + qOff;
							double k2 = latt->GGT.metric_length_squared(wrap(k));
							if (k2 < k2thr){
								nearValley = true; break;
							}
						}
						if (nearValley) break;
					}
					if (nearValley) break;
				}
				if (!nearValley) { oskip = oskip + 1; continue; }
			}

			fw.eLoop(k0[o] + qOff, electron::kSelect, this);
			otot = otot + 1;
		}
		if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%% ", int(round((o - oStart + 1)*100. / noMine))); logFlush(); } //Print progress:
	}
	logPrintf("done.\n"); logFlush();
	if (select_k_use_meff){
		otot = otot + oskip;
		mpiWorld->allReduce(oskip, MPIUtil::ReduceSum); mpiWorld->allReduce(otot, MPIUtil::ReduceSum);
		logPrintf("%lf qOff of %lf are skipped\n", oskip, otot);
	}

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

void electron::kpointSelect_scatt(){
	if (fw.isMetal || !assumeMetal_scatt){
		logPrintf("Unnecessary to change k points\n"); logFlush();
		return;
	}

	std::vector<FeynWann::StateE> states;
	set_mu(E); // random g factors needs mu
	generate_states_elec(states);
	for (int ik = 0; ik < k.size(); ik++)
		std::copy(states[ik].E.begin(), states[ik].E.end(), E.begin() + ik * nBandsSel_probe);
	report_density(states);
	
	//reset energy range
	double Emargin = nkBT*Tmax; //neglect below 10^-3 occupation deviation from equilibrium
	EBot_eph = dmu - pumpOmegaMax - std::max(5. / pumpTau + 3.*Tmax, Emargin);
	ETop_eph = dmu + pumpOmegaMax + std::max(5. / pumpTau + 3.*Tmax, Emargin);
	logPrintf("Active energy range for kpointSelect: %.3lf to %.3lf eV\n", EBot_eph / eV, ETop_eph / eV);

	if (read_kpts_2nd){
		saveEk("morek");

		std::vector<vector3<>>().swap(k);
		std::vector<double>().swap(E);

		logPrintf("\nRead k points from ldbd_data/ldbd_kvec.bin.\n"); logFlush();
		assert(exists("ldbd_data/ldbd_size.dat"));
		FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
		char s[200]; fgets(s, sizeof s, fp); fgets(s, sizeof s, fp);
		if (fgets(s, sizeof s, fp) != NULL){
			double dtmp; int nk; sscanf(s, "%le %d", &dtmp, &nk);
			k.resize(nk, vector3<>(0, 0, 0)); E.resize(nk*nBandsSel_probe); logPrintf("number of k = %lu\n", k.size());
		}
		fclose(fp);

		assert(exists("ldbd_data/ldbd_kvec.bin")); assert(exists("ldbd_data/ldbd_ek.bin"));
		FILE *fpk = fopen("ldbd_data/ldbd_kvec.bin", "rb"), *fpe = fopen("ldbd_data/ldbd_ek.bin", "rb");
		size_t expected_size_k = k.size() * 3 * sizeof(double), expected_size_e = k.size() * nBandsSel_probe * sizeof(double);
		check_file_size(fpk, expected_size_k, "ldbd_kvec.bin size does not match expected size"); check_file_size(fpe, expected_size_e, "ldbd_ek.bin size does not match expected size");
		fread(k.data(), sizeof(double), k.size() * 3, fpk);
		fread(E.data(), sizeof(double), k.size() * nBandsSel_probe, fpe);
		fclose(fpk); fclose(fpe);

		return;
	}

	//select k points having energies within [new EBot_eph, new ETop_eph]
	logPrintf("\nSelect k points for scattering\n"); logFlush();
	std::vector<vector3<>> ktmp(k);
	std::vector<double> Etmp(E);
	std::vector<vector3<>> ktmp2;
	std::vector<double> Etmp2;
	std::vector<vector3<>>().swap(k);
	std::vector<double>().swap(E);

	for (int ik = 0; ik < ktmp.size(); ik++){
		bool inrange = false;
		for (int b = 0; b < nBandsSel_probe; b++){
			if (Etmp[ik*nBandsSel_probe + b] >= EBot_eph && Etmp[ik*nBandsSel_probe + b] <= ETop_eph){
				inrange = true; break;
			}
		}
		if (inrange){
			k.push_back(ktmp[ik]);
			E.insert(E.end(), Etmp.begin() + ik*nBandsSel_probe, Etmp.begin() + (ik + 1)*nBandsSel_probe);
		}
		else{
			ktmp2.push_back(ktmp[ik]);
			Etmp2.insert(Etmp2.end(), Etmp.begin() + ik*nBandsSel_probe, Etmp.begin() + (ik + 1)*nBandsSel_probe);
		}
	}
	std::vector<vector3<>>().swap(ktmp);
	std::vector<double>().swap(Etmp);
	// ktmp = k + ktmp2 is reorder of initial k
	ktmp.insert(ktmp.end(), k.begin(), k.end());
	ktmp.insert(ktmp.end(), ktmp2.begin(), ktmp2.end());
	Etmp.insert(Etmp.end(), E.begin(), E.end());
	Etmp.insert(Etmp.end(), Etmp2.begin(), Etmp2.end());

	//write reordered k points having energies within [old Ebot_eph, old Ebot_eph]
	k.swap(ktmp);
	E.swap(Etmp);
	logPrintf("size of reordered vector k: %lu\n", k.size()); logFlush();
	logPrintf("size of reordered vector E: %lu\n", E.size()); logFlush();

	saveEk("morek");

	//let k store k points having energies within [new EBot_eph, new ETop_eph]
	k.swap(ktmp);
	E.swap(Etmp);
	logPrintf("number of k points for scattering: %lu\n", k.size()); logFlush();
	logPrintf("size of vector E for scattering: %lu\n", E.size()); logFlush();

	saveEk();
}
