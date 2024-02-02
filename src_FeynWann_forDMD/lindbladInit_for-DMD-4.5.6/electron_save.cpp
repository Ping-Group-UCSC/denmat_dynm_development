#include "electron.h"

void electron::saveR(){
	if (mpiWorld->isHead()){
		//int dim = 3;
		//for (int idir = 0; idir < 3; idir++)
		//	if (fw.isTruncated[idir]) dim -= 1;
		string fname = dir_ldbd + "ldbd_R.dat";
		FILE *fp = fopen(fname.c_str(), "w");
		//fprintf(fp, "%d\n", dim);
		fprintf(fp, "%d\n", latt->dimension());
		fw.R.print(fp, " %14.7le", false);
		fclose(fp);
	}
}
void electron::saveSize(parameters *param, size_t nkpairs, double omegaMax){
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
			fprintf(fp, "%d %d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph_elec bTop_eph_elec nb_wannier bBot_probe band_dft_skipped\n",
			bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, bBot_eph - bRef, bTop_eph - bRef, fw.nBands, bRef, band_skipped);
		else if (eScattOnlyHole)
			fprintf(fp, "%d %d %d %d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph_elec bTop_eph_elec bBot_eph_hole bTop_eph_hole nb_wannier bBot_probe band_dft_skipped\n",
			bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, 0, 0, bBot_eph - bRef, bTop_eph - bRef, fw.nBands, bRef, band_skipped);
		else
			fprintf(fp, "%d %d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph bTop_eph nb_wannier bBot_probe band_dft_skipped\n",
			bTop_probe - bRef, bCBM - bRef, bBot_dm - bRef, bTop_dm - bRef, bBot_eph - bRef, bTop_eph - bRef, fw.nBands, bRef, band_skipped);
		fprintf(fp, "%21.14le %lu %d %d %d # nk_full nk kmesh\n", nkTot, k.size(), NkFine[0], NkFine[1], NkFine[2]);
		if (eScattOnlyElec)
			fprintf(fp, "%lu # nkpair_elec\n", nkpairs);
		else if (eScattOnlyHole)
			fprintf(fp, "%d %lu # nkpair_elec nkpair_hole\n", 0, nkpairs);
		else
			fprintf(fp, "%lu # nkpair\n", nkpairs);
		fprintf(fp, "%d %d %d %d# modeStart modeStp modeSkipStart modeSkipStop\n", param->modeStart, param->modeStop, param->modeSkipStart, param->modeSkipStop);
		fprintf(fp, "%21.14le # T\n", Tmax); // Currently T = Tmax
		fprintf(fp, "%21.14le %21.14le %21.14le # muMin, muMax mu (given carrier density)\n", dmuMin, dmuMax, dmu);
		fprintf(fp, "%lg %lg # degauss\n", param->ePhDelta, param->nEphDelta);
		fprintf(fp, "%14.7le %14.7le %14.7le %14.7le %14.7le %14.7le # EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph\n", EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph);
		fprintf(fp, "%14.7le # omegaMax", omegaMax);
		fclose(fp);
	}
}
void electron::saveEk(string suffix){
	//Maybe formatted file is better
	if (mpiWorld->isHead()){
		string fname = dir_ldbd + "ldbd_kvec.bin";
		if (suffix != "") fname = dir_ldbd + "ldbd_kvec" + "_" + suffix + ".bin";
		FILE *fp = fopen(fname.c_str(), "wb");
		fwrite(k.data(), sizeof(double), k.size() * 3, fp);
		fclose(fp);
		fname = dir_ldbd + "ldbd_ek.bin";
		if (suffix != "") fname = dir_ldbd + "ldbd_ek" + "_" + suffix + ".bin";
		fp = fopen(fname.c_str(), "wb");
		fwrite(E.data(), sizeof(double), k.size() * nBandsSel_probe, fp);
		fclose(fp);
	}
}

void electron::generate_gfac(){
	if (gfack.size() == 0 && (gfac_sigma[0] > 0 || gfac_sigma[1] > 0 || gfac_sigma[2] > 0)){
		logPrintf("\ngenerate k-depenedent g-factors with normal distribution:\n");
		gfack.resize(k.size());

		if (mpiWorld->isHead()){
			random_normal_(gfack, matrix3<>(gfac_mean[0], gfac_mean[1], gfac_mean[2]), 
				matrix3<>(gfac_sigma[0], gfac_sigma[1], gfac_sigma[2]), 
				matrix3<>(gfac_cap[0], gfac_cap[1], gfac_cap[2]));
			matrix3<> mean_gfack = mean_of_(gfack);
			matrix3<> sigma_gfack = sigma_of_(gfack);
			logPrintf("mean of gfack without weights:\n"); logFlush();
			if (mpiWorld->isHead()) mean_gfack.print(stdout, " %lg"); logFlush();
			logPrintf("sigma of gfack without weights:\n"); logFlush();
			if (mpiWorld->isHead()) sigma_gfack.print(stdout, " %lg"); logFlush();

			std::vector<double> sum_dfde_k(k.size());
			for (size_t ik = 0; ik < k.size(); ik++){
				for (int b = 0; b < nBandsSel; b++){
					double f = fermi((E[ik*nBandsSel_probe + b + bStart - bRef] - dmu) / Tmax);
					sum_dfde_k[ik] += f * (1-f);
				}
			}
			
			random_normal_(gfack, matrix3<>(gfac_mean[0], gfac_mean[1], gfac_mean[2]), 
				matrix3<>(gfac_sigma[0], gfac_sigma[1], gfac_sigma[2]), 
				matrix3<>(gfac_cap[0], gfac_cap[1], gfac_cap[2]), sum_dfde_k.data());
			mean_gfack = mean_of_(gfack, sum_dfde_k.data());
			sigma_gfack = sigma_of_(gfack, false, mean_gfack, sum_dfde_k.data());
			logPrintf("mean of gfack with weights:\n"); logFlush();
			if (mpiWorld->isHead()) mean_gfack.print(stdout, " %lg"); logFlush();
			logPrintf("sigma of gfack with weights:\n"); logFlush();
			if (mpiWorld->isHead()) sigma_gfack.print(stdout, " %lg"); logFlush();
			fprintf("gfack_normal.out", gfack, "#gfack:", " %14.10lf");
		}

		mpiWorld->bcastData(gfack);
	}
}
void electron::generate_Bso(){
	if (Bsok.size() == 0 and fw.fwp.Bin_model != "none" and fw.fwp.Bin_model != "read" and fabs(alpha_Bin) > 0){
		logPrintf("\ngenerate k-depenedent model SOC fields\n");
		Bsok.resize(k.size());

		if (mpiWorld->isHead()){
			std::vector<double> sum_dfde_k(k.size());
			for (size_t ik = 0; ik < k.size(); ik++){
				Bsok[ik] = fw.calc_Bso(k[ik]);

				for (int b = 0; b < nBandsSel; b++){
					double f = fermi((E[ik*nBandsSel_probe + b + bStart - bRef] - dmu) / Tmax);
					sum_dfde_k[ik] += f * (1 - f);
				}
			}

			vector3<> mean_Bsok = mean_of_(Bsok, sum_dfde_k.data());
			vector3<> sigma_Bsok = sigma_of_(Bsok, false, mean_Bsok, sum_dfde_k.data());
			logPrintf("mean of Bsok with weights: %lg %lg %lg\n", mean_Bsok[0], mean_Bsok[1], mean_Bsok[2]); logFlush();
			logPrintf("sigma of Bsok with weights: %lg %lg %lg\n", sigma_Bsok[0], sigma_Bsok[1], sigma_Bsok[2]); logFlush();
			fprintf("Bsok_model.out", Bsok, "#Bsok:", " %14.10lf");
		}

		mpiWorld->bcastData(Bsok);
	}
}

void electron::analyse_gfac(){
	if (mpiWorld->isHead() && gfack.size() > 0){
		std::vector<double> sum_dfde_k(k.size());
		for (size_t ik = 0; ik < k.size(); ik++)
		for (int b = 0; b < nBandsSel; b++)
			sum_dfde_k[ik] += F[ik][b] * (1 - F[ik][b]);

		matrix3<> mean_gfack = mean_of_(gfack, sum_dfde_k.data());
		matrix3<> sigma_gfack = sigma_of_(gfack, false, mean_gfack, sum_dfde_k.data());
		logPrintf("mean of gfack without weights:\n"); logFlush();
		mean_gfack.print(stdout, " %lg"); logFlush();
		logPrintf("sigma of gfack without weights:\n"); logFlush();
		sigma_gfack.print(stdout, " %lg"); logFlush();
	}
}
void electron::analyse_Bso(){
	if (mpiWorld->isHead() && Bsok.size() > 0){
		std::vector<double> sum_dfde_k(k.size());
		for (size_t ik = 0; ik < k.size(); ik++)
		for (int b = 0; b < nBandsSel; b++)
			sum_dfde_k[ik] += F[ik][b] * (1 - F[ik][b]);

		vector3<> mean_Bsok = mean_of_(Bsok, sum_dfde_k.data());
		vector3<> sigma_Bsok = sigma_of_(Bsok, false, mean_Bsok, sum_dfde_k.data());
		logPrintf("mean of Bsok with weights: %lg %lg %lg\n", mean_Bsok[0], mean_Bsok[1], mean_Bsok[2]); logFlush();
		logPrintf("sigma of Bsok with weights: %lg %lg %lg\n", sigma_Bsok[0], sigma_Bsok[1], sigma_Bsok[2]); logFlush();
	}
}

void electron::set_mu(std::vector<double>& E){
	// compute Fermi-Dirac occupations and carrier densities
	if (n_dmu == 0 || fw.isMetal)
		dmu = dmuMax;
	else{
		double ncarrier = n_dmu * latt->cell_size() * nkTot;
		dmu = find_mu(ncarrier, Tmax, n_dmu > 0 ? EcMin : EvMax, E, k.size(), bStart - bRef, bCBM - bRef, bStop - bRef);
	}
	logPrintf("\ndmu = %21.14le (%21.14le eV)\n", dmu, dmu / eV); logFlush();
}
void electron::set_mu(std::vector<FeynWann::StateE>& states){
	// compute Fermi-Dirac occupations and carrier densities
	if (n_dmu == 0 || fw.isMetal)
		dmu = dmuMax;
	else{
		double ncarrier = n_dmu * latt->cell_size() * nkTot;
		dmu = find_mu(ncarrier, Tmax, n_dmu > 0 ? EcMin : EvMax, states, bStart - bRef, bCBM - bRef, bStop - bRef);
	}
	logPrintf("\ndmu = %21.14le (%21.14le eV)\n", dmu, dmu / eV); logFlush();
}
void electron::report_density(std::vector<FeynWann::StateE>& states){
	double n_maj_dmuMax = compute_ncarrier(dmuMax < Emid, Tmax, dmuMax, states, bStart - bRef, bCBM - bRef, bStop - bRef) / nkTot / latt->cell_size();
	double n_min_dmuMax = compute_ncarrier(dmuMax >= Emid, Tmax, dmuMax, states, bStart - bRef, bCBM - bRef, bStop - bRef) / nkTot / latt->cell_size();
	double n_maj_dmuMin = compute_ncarrier(dmuMin < Emid, Tmax, dmuMin, states, bStart - bRef, bCBM - bRef, bStop - bRef) / nkTot / latt->cell_size();
	double n_min_dmuMin = compute_ncarrier(dmuMin >= Emid, Tmax, dmuMin, states, bStart - bRef, bCBM - bRef, bStop - bRef) / nkTot / latt->cell_size();
	logPrintf("\nAt dmuMax, majority density: "); latt->print_carrier_density(n_maj_dmuMax); logFlush();
	logPrintf("At dmuMax, minority density: "); latt->print_carrier_density(n_min_dmuMax); logFlush();
	logPrintf("At dmuMin, majority density: "); latt->print_carrier_density(n_maj_dmuMin); logFlush();
	logPrintf("At dmuMin, minority density: "); latt->print_carrier_density(n_min_dmuMin); logFlush();
}

void electron::generate_states_elec(std::vector<FeynWann::StateE>& states){
	generate_gfac();
	generate_Bso();

	TaskDivision tasks(k.size(), mpiWorld);
	size_t ikStart, ikStop;
	tasks.myRange(ikStart, ikStop);
	FeynWann::StateE eTrunc;
	{ FeynWann::StateE e; fw.eCalc(k[ikStart], e); fw.trunc_stateE(e, eTrunc, bBot_eph, bTop_eph, bBot_dm, bTop_dm, bBot_probe, bTop_probe); }
	states.resize(k.size(), eTrunc);
	int kInterval = std::max(1, int(round(k.size() / 50.)));
	logPrintf("\nCompute StateE in parallel:"); logFlush();
	for (size_t ik = ikStart; ik < ikStop; ik++){
		if (gfack.size() > 0) fw.gfac = gfack[ik];
		if (Bsok.size() > 0) fw.Bso = Bsok[ik];
		{// e will be free after "}"
			FeynWann::StateE e; fw.eCalc(k[ik], e);
			fw.trunc_stateE(e, states[ik], bBot_eph, bTop_eph, bBot_dm, bTop_dm, bBot_probe, bTop_probe);
			if (!fw.energyOnly && save_dHePhSum_disk && e.dHePhSum.nRows() == fw.nBands*fw.nBands && e.dHePhSum.nCols() == 3){
				//write dHePhSum to file
				string fname = "ldbd_data/dHePhSum/k" + int2str(ik) + ".bin";
				FILE *fp = fopen(fname.c_str(), "wb");
				fwrite(e.dHePhSum.data(), 2 * sizeof(double), fw.nBands*fw.nBands * 3, fp);
				fclose(fp);
			}
		}
		if ((ik + 1) % kInterval == 0) { logPrintf("%d%% ", int(round((ik + 1)*100. / k.size()))); logFlush(); }
	}
	logPrintf(" done\n"); logFlush();

	MPI_Barrier(MPI_COMM_WORLD);
	for (int whose = 0; whose < mpiWorld->nProcesses(); whose++)
	for (size_t ik = tasks.start(whose); ik < tasks.stop(whose); ik++)
		fw.bcastState_JX(states[ik], mpiWorld, whose, !save_dHePhSum_disk);
	logPrintf("bcastState_JX done\n");

	logPrintf("\nsize of E: %lu\n", states[ikStart].E.size());
	logPrintf("size of U: %d %d\n", states[ikStart].U.nRows(), states[ikStart].U.nCols());
	logPrintf("size of S: %d %d\n", states[ikStart].S[2].nRows(), states[ikStart].S[2].nCols());
	logPrintf("size of L: %d %d\n", states[ikStart].L[2].nRows(), states[ikStart].L[2].nCols());
	logPrintf("size of v: %d %d\n\n", states[ikStart].v[2].nRows(), states[ikStart].v[2].nCols());
	MPI_Barrier(MPI_COMM_WORLD);
}

void electron::saveElec(){
	MPI_Barrier(MPI_COMM_WORLD);
	if (mpiWorld->isHead()){
		//  write energies
		string fnamee = dir_ldbd + "ldbd_ek.bin";
		FILE *fpe = fopen(fnamee.c_str(), "wb");
		for (size_t ik = 0; ik < k.size(); ik++){
			//diagMatrix E = state_elec[ik].E(bBot_probe, bTop_probe); // notice the band range here
			//fwrite(E.data(), sizeof(double), nBandsSel_probe, fpe);
			fwrite(state_elec[ik].E.data(), sizeof(double), nBandsSel_probe, fpe);
		}
		fclose(fpe);
	}
	if (mpiWorld->isHead() && fw.fwp.needSpin){
		//  write spin matrices
		string fnames = dir_ldbd + "ldbd_smat.bin";
		FILE *fps = fopen(fnames.c_str(), "wb");
		for (size_t ik = 0; ik < k.size(); ik++)
		for (size_t iDir = 0; iDir < 3; iDir++){
			//matrix s = 0.5*state_elec[ik].S[iDir](bBot_dm, bTop_dm, bBot_dm, bTop_dm); // notice the band range here
			matrix s = 0.5*state_elec[ik].S[iDir];
			matrix st = transpose(s); // from ColMajor to RowMajor
			fwrite(st.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*(bTop_dm - bBot_dm), fps);
		}
		fclose(fps);
	}
	if (mpiWorld->isHead() && fw.fwp.needL){
		//  write L matrices
		string fnamel = dir_ldbd + "ldbd_lmat.bin";
		FILE *fpl = fopen(fnamel.c_str(), "wb");
		for (size_t ik = 0; ik < k.size(); ik++)
		for (size_t iDir = 0; iDir < 3; iDir++){
			matrix l = state_elec[ik].L[iDir];
			matrix lt = transpose(l); // from ColMajor to RowMajor
			fwrite(lt.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*(bTop_dm - bBot_dm), fpl);
		}
		fclose(fpl);
	}
	if (mpiWorld->isHead() && layerOcc){
		//  write layerDensity matrices
		string fnamel = dir_ldbd + "ldbd_layermat.bin", fnamesl = dir_ldbd + "ldbd_spinlayermat.bin", fnamels = dir_ldbd + "ldbd_layerspinmat.bin";
		FILE *fpl = fopen(fnamel.c_str(), "wb"), *fpsl = fopen(fnamesl.c_str(), "wb"), *fpls = fopen(fnamels.c_str(), "wb");
		for (size_t ik = 0; ik < k.size(); ik++){
			//matrix l = state_elec[ik].layer(bBot_dm, bTop_dm, bBot_dm, bTop_dm); // notice the band range here
			matrix l = state_elec[ik].layer;
			matrix lt = transpose(l); // from ColMajor to RowMajor
			fwrite(lt.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*(bTop_dm - bBot_dm), fpl);

			//matrix slfull = state_elec[ik].S[2] * state_elec[ik].layer; // notice the band range here
			//matrix sl = 0.5*slfull(bBot_dm, bTop_dm, bBot_dm, bTop_dm);
			matrix sl = state_elec[ik].S[2] * state_elec[ik].layer;
			matrix slt = transpose(sl); // from ColMajor to RowMajor
			fwrite(slt.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*(bTop_dm - bBot_dm), fpsl);

			//matrix lsfull = state_elec[ik].layer * state_elec[ik].S[2]; // notice the band range here
			//matrix ls = 0.5*lsfull(bBot_dm, bTop_dm, bBot_dm, bTop_dm);
			matrix ls = state_elec[ik].layer * state_elec[ik].S[2];
			matrix lst = transpose(ls); // from ColMajor to RowMajor
			fwrite(lst.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*(bTop_dm - bBot_dm), fpls);
		}
		fclose(fpl); fclose(fpsl); fclose(fpls);
	}
	if (mpiWorld->isHead()){
		string fnamevmat = dir_ldbd + "ldbd_vmat.bin", fnamevvec = dir_ldbd + "ldbd_vvec.bin";
		FILE *fpvmat = fopen(fnamevmat.c_str(), "wb"), *fpvvec = fopen(fnamevvec.c_str(), "wb");
		for (size_t ik = 0; ik < k.size(); ik++){
			for (size_t iDir = 0; iDir < 3; iDir++){
				//matrix v = state_elec[ik].v[iDir](bBot_dm, bTop_dm, bBot_probe, bTop_probe); // notice the band range here
				matrix v = state_elec[ik].v[iDir];
				matrix vt = transpose(v); // from ColMajor to RowMajor
				fwrite(vt.data(), 2 * sizeof(double), (bTop_dm - bBot_dm)*nBandsSel_probe, fpvmat);
			}
			//for (int b = bBot_eph; b < bTop_eph; b++)
			for (int b = 0; b < bTop_eph - bBot_eph; b++)
				fwrite(&state_elec[ik].vVec[b][0], sizeof(double), 3, fpvvec);
		}
		fclose(fpvmat); fclose(fpvvec);
	}
	if (mpiWorld->isHead() && writeU){
		string fnameu = dir_ldbd + "ldbd_Umat.bin";
		FILE *fpu = fopen(fnameu.c_str(), "wb");
		for (size_t ik = 0; ik < k.size(); ik++){
			//matrix U = state_elec[ik].U; // full U matrix
			matrix U = state_elec[ik].U;
			matrix Ut = transpose(U); // from ColMajor to RowMajor
			//fwrite(Ut.data(), 2 * sizeof(double), fw.nBands*fw.nBands, fpu);
			fwrite(Ut.data(), 2 * sizeof(double), fw.nBands*(bTop_eph - bBot_eph), fpu);
		}
		fclose(fpu);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void electron::savekData(parameters *param, size_t nkpairs, double omegaMax){
	// please never use -G? when running this program
	if (mpiWorld->isHead() && save_dHePhSum_disk) system("mkdir ldbd_data/dHePhSum");

	if (!read_kpts){
		saveR(); logPrintf("saveR done\n"); logFlush();
		saveSize(param, nkpairs, omegaMax); logPrintf("saveSize done\n"); logFlush();
		saveEk(); logPrintf("saveEk done\n"); logFlush();
	}

	// main subroutine: generate electronic states and save electronic quantities
	if (fw.isMetal || !assumeMetal_scatt) set_mu(E); // random g factors needs mu
	generate_states_elec(state_elec);
	if (fw.isMetal || !assumeMetal_scatt) report_density(state_elec);
	saveElec(); logPrintf("saveElec done\n"); logFlush();

	F = computeF(Tmax, dmu, state_elec, bStart - bRef, bStop - bRef);
	// g factor and Bso
	analyse_gfac();
	analyse_Bso();

	// spin mixing
	if (mpiWorld->isHead()){
		bsq = compute_bsq();
		vector3<> bsq_avg; average_dfde(F, bsq, bsq_avg);
		logPrintf("\nSpin mixing |b|^2: %lg %lg %lg\n", bsq_avg[0], bsq_avg[1], bsq_avg[2]); logFlush();
	}
}