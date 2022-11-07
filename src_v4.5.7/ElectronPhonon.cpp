#include "ElectronPhonon.h"

void electronphonon::set_eph(){
	alloc_ephmat(mp->varstart, mp->varend); // allocate matrix A or P
	set_kpair();
	if (alg.linearize) return;
	set_ephmat();
	compute_imsig("eph");
	for (int iD = 0; iD < eip.ni.size(); iD++){ if (!alg.only_ee) add_scatt_contrib("eimp", iD); }
	if (eip.ni.size() >0 && !alg.only_ee) compute_imsig("eph_eimp");
	//if (ee_model != nullptr && eep.eeMode == "Pee_fixed_at_eq" && !alg.only_eimp) { add_scatt_contrib("ee"); compute_imsig("eph_eimp_ee"); }
	if (ee_model != nullptr && !alg.only_eimp) { add_scatt_contrib("ee"); compute_imsig("eph_eimp_ee"); } // to have ImSigma and mobility due to e-e, this should be always called
	set_sparseP(true);
}
void electronphonon::reset_scatt(bool reset_eimp, bool reset_ee, complex **dm_expand, complex **dm1_expand, double t, double **f_eq_expand){
	// currently reset_eimp || reset_ee means resetting both e-i and e-e scatterings
	// if you want to reset only one of them, you will need to modify the code to store another
	if ((!reset_eimp && !reset_ee) || (eimp == nullptr && ee_model == nullptr)) return;
	trunc_copy_arraymat(dm, dm_expand, nk_glob, nb_expand, bStart, bEnd);
	if (dm1_expand != nullptr)
		trunc_copy_arraymat(dm1, dm1_expand, nk_glob, nb_expand, bStart, bEnd);
	else{
		axbyc(dm1, dm, nk_glob, nb*nb, cm1); // dm1 = -dm
		for (int ik = 0; ik < nk_glob; ik++)
		for (int i = 0; i < nb; i++)
			dm1[ik][i*nb + i] += c1;
	}
	if (alg.linearize_dPee){
		trunc_copy_array(f_eq, f_eq_expand, nk_glob, bStart, bEnd);
		axbyc(f1_eq, f_eq, nk_glob, nb, -1, 0, 1);
	}

	set_ephmat();
	if (coul_model != nullptr && clp.update) coul_model->init(dm);
	for (int iD = 0; iD < eip.ni.size(); iD++)
		if (!alg.only_ee){ 
			if (eimp[iD]->eimp_model != nullptr && eimp[iD]->eimp_model->imsig != nullptr) zeros(eimp[iD]->eimp_model->imsig, nk_glob, nb); 
			add_scatt_contrib("eimp", iD); 
		}
	if (ee_model != nullptr && !alg.only_eimp) { if (ee_model->imsig != nullptr) zeros(ee_model->imsig, nk_glob, nb); add_scatt_contrib("ee", 0, dm, dm1, t); }
	set_sparseP(false);
}

void electronphonon::add_scatt_contrib(string what, int iD, complex **dm, complex **dm1, double t){
	ostringstream convert; convert << mp->myrank;
	convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
	FILE *fp; string fname = dir_debug + "debug_add_P" + what + ".out." + convert.str();
	if (exists(fname)) fname = dir_debug + "debug_add_P" + what + "_updated.out." + convert.str();
	bool ldebug = DEBUG;
	if (ldebug) fp = fopen(fname.c_str(), "a");
	MPI_Barrier(MPI_COMM_WORLD);

	if (ionode && what == "eimp") printf("\nAdd electron-impurity (%d) scattering contribution to P?\n", iD);
	if (ionode && what == "ee") printf("\nAdd electron-electron scattering contribution to P?\n");
	complex *P1add = new complex[(int)std::pow(nb, 4)]; complex *P2add = new complex[(int)std::pow(nb, 4)];

	// we need density matrix in schrodinger picture for e-e scattering
	if (what == "ee" && dm != nullptr && alg.picture == "interaction"){
		for (int ik = 0; ik < nk_glob; ik++)
		for (int i = 0; i < nb; i++)
		for (int j = i + 1; j < nb; j++){
			dm[ik][i*nb + j] *= cis((e[ik][j] - e[ik][i])*t); // from interaction to schrodinger
			dm1[ik][i*nb + j] = -dm[ik][i*nb + j];
			dm[ik][j*nb + i] = conj(dm[ik][i*nb + j]);
			dm1[ik][j*nb + i] = conj(dm1[ik][i*nb + j]);
		}
	}

	double nk3_accum = 0;
	double scale_fac = scale_scatt;
	if (what == "eimp") scale_fac = scale_ei;
	if (what == "ee") scale_fac = scale_ee;
	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];
		if (ldebug) { fprintf(fp, "\nikpair=%d(%d) k1=%d k2=%d\n", ikpair_local, nkpair_proc, ik_glob, ikp_glob); fflush(fp); }
		zeros(P1add, (int)std::pow(nb, 4)); zeros(P2add, (int)std::pow(nb, 4));

		if (what == "eimp"){
			if (eip.impMode[iD] == "ab_neutral") eimp[iD]->read_ldbd_imp_P(ikpair_local, P1add, P2add);
			if (eip.impMode[iD] == "model_ionized" && !ldebug) eimp[iD]->eimp_model->calc_P(ik_glob, ikp_glob, P1add, P2add, false);
			if (eip.impMode[iD] == "model_ionized" && ldebug) eimp[iD]->eimp_model->calc_P_debug(ik_glob, ikp_glob, P1add, P2add, false);
		}
		if (what == "ee" && alg.linearize_dPee) nk3_accum += ee_model->calc_P(ik_glob, ikp_glob, P1add, P2add, false, dm, dm1, dP1ee[ikpair_local], dP2ee[ikpair_local], f_eq, f1_eq);
		if (what == "ee" && !alg.linearize_dPee) nk3_accum += ee_model->calc_P(ik_glob, ikp_glob, P1add, P2add, false, dm, dm1);

		if (scale_fac != 1.) axbyc(P1add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));
		if (scale_fac != 1.) axbyc(P2add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));

		complex factor = ((what == "eimp" && iD == 0 && alg.only_eimp) || (what == "ee" && alg.only_ee)) ? c0 : c1;
		if (alg.Pin_is_sparse){
			sparse_plus_dense(sP1->smat[ikpair_local], alg.thr_sparseP, P1add, nb*nb, nb*nb, c1, factor);
			sparse_plus_dense(sP2->smat[ikpair_local], alg.thr_sparseP, P2add, nb*nb, nb*nb, c1, factor);
		}
		else{
			axbyc(P1[ikpair_local], P1add, (int)std::pow(nb, 4), c1, factor);
			axbyc(P2[ikpair_local], P2add, (int)std::pow(nb, 4), c1, factor);
		}
		if (ldebug){
			fprintf_complex_mat(fp, P1add, nb*nb, "P1add:"); fflush(fp);
			fprintf_complex_mat(fp, P2add, nb*nb, "P2add:"); fflush(fp);
		}
		if (ionode && what == "ee" && ikpair_local % 1000 == 0)  printf("kpair %d done\n", ikpair_local);
	}
	delete[] P1add; delete[] P2add;
	double max = nk3_accum, min = nk3_accum, avg = nk3_accum;
	mp->allreduce(max, MPI_MAX); mp->allreduce(min, MPI_MIN); mp->allreduce(avg, MPI_SUM); avg /= mp->nprocs;
	if (what == "ee" && ionode) printf("nk3_accum: max= %lg min= %lg avg = %lg\n", max, min, avg);

	if (what == "eimp" && eip.impMode[iD] == "model_ionized") eimp[iD]->eimp_model->reduce_imsig(this->mp);
	if (what == "ee") ee_model->reduce_imsig(this->mp);
	if (ldebug) fclose(fp);
}

void electronphonon::set_sparseP(bool fisrt_call){
	if (alg.Pin_is_sparse) return;
	if (!alg.sparseP && !fisrt_call) return;
	if (sP1 != nullptr) { delete sP1; sP1 = nullptr; }
	if (sP2 != nullptr) { delete sP2; sP2 = nullptr; }
	sP1 = new sparse2D(mp, P1, nb*nb, nb*nb, alg.thr_sparseP);
	sP2 = new sparse2D(mp, P2, nb*nb, nb*nb, alg.thr_sparseP);
	if (alg.sparseP){
		if (ionode) printf("Construct sP1 and sP2 from P1 and P2\n");
		sP1->sparse(P1, false); // do_test = false
		sP2->sparse(P2, false);
		string suffix = isHole ? alg.scatt + "_hole" : alg.scatt;
		sP1->write_smat("ldbd_data/sP1_" + suffix + "_ns.bin", "ldbd_data/sP1_" + suffix + "_s.bin", "ldbd_data/sP1_" + alg.scatt + "_i.bin", "ldbd_data/sP1_" + alg.scatt + "_j.bin");
		sP2->write_smat("ldbd_data/sP2_" + suffix + "_ns.bin", "ldbd_data/sP2_" + suffix + "_s.bin", "ldbd_data/sP2_" + alg.scatt + "_i.bin", "ldbd_data/sP2_" + alg.scatt + "_j.bin");
	}
}

void electronphonon::get_brange(bool sepr_eh, bool isHole){
	if (ionode) printf("\nread ldbd_size.dat to get band range:\n");
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		int itmp1, itmp2, itmp3, itmp4, itmp5, itmp6;
		if (!isHole)
			sscanf(s, "%d %d %d %d %d %d", &itmp1, &itmp2, &itmp3, &itmp4, &bStart, &bEnd);
		else
			sscanf(s, "%d %d %d %d %d %d %d %d", &itmp1, &itmp2, &itmp3, &itmp4, &itmp5, &itmp6, &bStart, &bEnd);
		bStart -= elec->bStart_dm; bEnd -= elec->bStart_dm;
		nv = std::max(elec->nv_dm - bStart, 0);
		nc = std::max(bEnd - elec->nv_dm, 0);
		if (ionode) printf("bStart = %d bEnd = %d\n", bStart, bEnd);
	}
	nb = bEnd - bStart;

	for (int i = 0; i < 6; i++) { fgets(s, sizeof s, fp); }
	if (fgets(s, sizeof s, fp) != NULL){
		double d1, d2, d3, d4;
		sscanf(s, "%lg %lg %lg %lg %lg %lg", &d1, &d2, &d3, &d4, &eStart, &eEnd);
	}
	if (bStart >= elec->nv_dm) eStart = elec->ecmin;
	if (bEnd <= elec->nv_dm) eEnd = elec->evmax;
	if (ionode) printf("estart_eph = %lg eend_eph = %lg\n", eStart, eEnd);
	fclose(fp);
}
void electronphonon::get_nkpair(){
	if (ionode) printf("\nread ldbd_size.dat to get nkpair_glob:\n");
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	fgets(s, sizeof s, fp);
	fgets(s, sizeof s, fp);
	fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		if (!isHole){
			sscanf(s, "%d", &nkpair_glob); if (ionode) printf("nkpair_glob = %d\n", nkpair_glob);
		}
		else{
			int itmp;
			sscanf(s, "%d %d", &itmp, &nkpair_glob); if (ionode) printf("nkpair_glob = %d\n", nkpair_glob);
		}
	}
	fclose(fp);
}
void electronphonon::set_kpair(){
	if (code == "jdftx")
		read_ldbd_kpair();
}
void electronphonon::read_ldbd_kpair(){
	if (ionode) printf("\nread ldbd_kpair_k1st(2nd)(_hole).bin:\n");
	FILE *fpk, *fpkp;
	if (!isHole){
		fpk = fopen("ldbd_data/ldbd_kpair_k1st.bin", "rb");
		fpkp = fopen("ldbd_data/ldbd_kpair_k2nd.bin", "rb");
	}
	else{
		fpk = fopen("ldbd_data/ldbd_kpair_k1st_hole.bin", "rb");
		fpkp = fopen("ldbd_data/ldbd_kpair_k2nd_hole.bin", "rb");
	}
	size_t expected_size = nkpair_glob*sizeof(size_t);
	check_file_size(fpk, expected_size, "ldbd_kpair_k1st(_hole).bin size does not match expected size");
	check_file_size(fpkp, expected_size, "ldbd_kpair_k2nd(_hole).bin size does not match expected size");

	int pos = ikpair0_glob * sizeof(size_t);
	fseek(fpk, pos, SEEK_SET);
	fseek(fpkp, pos, SEEK_SET);

	fread(k1st, sizeof(size_t), nkpair_proc, fpk);
	fread(k2nd, sizeof(size_t), nkpair_proc, fpkp);
	fclose(fpk); fclose(fpkp);
}

void electronphonon::set_ephmat(){
	if (code == "jdftx") read_ldbd_eph();
}
void electronphonon::read_ldbd_eph(){
	if (!alg.Pin_is_sparse){
		if (!alg.eph_enable) return;

		if (ionode) printf("\nread ldbd_P1(2)_(lindblad/conventional)(_hole).bin:\n");
		string fname1 = "ldbd_data/ldbd_P1_", fname2 = "ldbd_data/ldbd_P2_", suffix;
		suffix = isHole ? alg.scatt + "_hole" : alg.scatt;
		fname1 += (suffix + ".bin"); fname2 += (suffix + ".bin");

		size_t Psize = (size_t)std::pow(nb, 4);
		size_t expected_size = nkpair_glob*Psize * 2 * sizeof(double);

		FILE *fp1 = fopen(fname1.c_str(), "rb");
		check_file_size(fp1, expected_size, fname1 + " size does not match expected size");
		fseek_bigfile(fp1, ikpair0_glob, Psize * 2 * sizeof(double));
		fread(P1[0], 2 * sizeof(double), nkpair_proc * Psize, fp1);
		if (scale_eph != 1.) axbyc(P1, nullptr, nkpair_proc, Psize, c0, complex(scale_eph, 0));
		fclose(fp1);

		FILE *fp2 = fopen(fname2.c_str(), "rb");
		check_file_size(fp2, expected_size, fname1 + " size does not match expected size");
		fseek_bigfile(fp2, ikpair0_glob, Psize * 2 * sizeof(double));
		fread(P2[0], 2 * sizeof(double), nkpair_proc * Psize, fp2);
		if (scale_eph != 1.) axbyc(P2, nullptr, nkpair_proc, Psize, c0, complex(scale_eph, 0));
		fclose(fp2);
	}
	else{
		if (ionode) printf("Read sP1 and sP2\n");
		if (sP1 != nullptr) { delete sP1; sP1 = nullptr; }
		if (sP2 != nullptr) { delete sP2; sP2 = nullptr; }
		if (alg.eph_enable){
			string suffix = isHole ? alg.scatt + "_hole" : alg.scatt;
			sP1 = new sparse2D(mp, "ldbd_data/sP1_" + suffix + "_ns.bin", "ldbd_data/sP1_" + suffix + "_s.bin", "ldbd_data/sP1_" + suffix + "_i.bin", "ldbd_data/sP1_" + suffix + "_j.bin", nb*nb, nb*nb);
			sP2 = new sparse2D(mp, "ldbd_data/sP2_" + suffix + "_ns.bin", "ldbd_data/sP2_" + suffix + "_s.bin", "ldbd_data/sP2_" + suffix + "_i.bin", "ldbd_data/sP2_" + suffix + "_j.bin", nb*nb, nb*nb);
			sP1->read_smat(false); // do_test = false
			sP2->read_smat(false);
			sparse_plus_dense(sP1->smat, sP1->thrsh, nullptr, sP1->nk, sP1->ni, sP1->nj, c0, complex(scale_eph, 0));
			sparse_plus_dense(sP2->smat, sP2->thrsh, nullptr, sP2->nk, sP2->ni, sP2->nj, c0, complex(scale_eph, 0));
		}
		else{
			sP1 = new sparse2D(mp, nullptr, nb*nb, nb*nb, alg.thr_sparseP);
			sP2 = new sparse2D(mp, nullptr, nb*nb, nb*nb, alg.thr_sparseP);
		}
	}
}

void electronphonon::alloc_ephmat(int ikpair0, int ikpair1){
	ikpair0_glob = ikpair0; ikpair1_glob = ikpair1; nkpair_proc = ikpair1 - ikpair0;
	k1st = new size_t[nkpair_proc]{0};
	k2nd = new size_t[nkpair_proc]{0};
	if (alg.summode){
		if (!alg.linearize){
			if (!alg.Pin_is_sparse) P1 = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
			if (!alg.Pin_is_sparse) P2 = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
			if (!alg.Pin_is_sparse && !alg.sparseP) P1_next = new complex[(int)std::pow(nb, 4)]{c0};
			if (!alg.Pin_is_sparse && !alg.sparseP) P2_next = new complex[(int)std::pow(nb, 4)]{c0};
			if (alg.linearize_dPee) dP1ee = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
			if (alg.linearize_dPee) dP2ee = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
		}
		else{
			Lscii = alloc_array(nk_glob, (int)std::pow(nb, 4));
			Lscij = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
			Lscji = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
			Lsct = new complex[(int)std::pow(nb, 4)]{c0};
		}
		//if (alg.expt)
		P1t = new complex[(int)std::pow(nb, 4)]{c0};
		//if (!alg.linearize && alg.expt)
		P2t = new complex[(int)std::pow(nb, 4)]{c0};
	}
	else{
		if (alg.scatt == "lindblad"){
			App = alloc_array(nkpair_proc, nm, nb*nb);
			Amm = alloc_array(nkpair_proc, nm, nb*nb);
			Apm = alloc_array(nkpair_proc, nm, nb*nb);
			Amp = alloc_array(nkpair_proc, nm, nb*nb);
		}
		else
			error_message("alg.scatt must be lindblad if not alg.summode");
	}

	dm = alloc_array(nk_glob, nb*nb);
	dm1 = alloc_array(nk_glob, nb*nb);
	ddmdt_eph = alloc_array(nk_glob, nb*nb);

	if (alg.Pin_is_sparse || alg.sparseP){
		if (!alg.linearize) sm1_next = new sparse_mat((int)std::pow(nb, 4), true);
		if (!alg.linearize) sm2_next = new sparse_mat((int)std::pow(nb, 4), true);
		//if (alg.expt){
		smat1_time = new sparse_mat((int)std::pow(nb, 4), true);
		//if (!alg.linearize)
		smat2_time = new sparse_mat((int)std::pow(nb, 4), true);
		make_map();
		//}
	}
}

void electronphonon::make_map(){
	ij2i = new int[nb*nb]();
	ij2j = new int[nb*nb]();
	int ij = 0;
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++){
		ij2i[ij] = i;
		ij2j[ij] = j;
		ij++;
	}
}