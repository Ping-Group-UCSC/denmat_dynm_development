#include "ElectronPhonon.h"

void electronphonon::set_Lsc(double **f_eq){
	trunc_copy_array(this->f_eq, f_eq, nk_glob, bStart, bEnd);

	if (!alg.eph_enable){
		zeros(Lscii, nk_glob, (int)std::pow(nb, 4));
		zeros(Lscij, nkpair_proc, (int)std::pow(nb, 4)); zeros(Lscji, nkpair_proc, (int)std::pow(nb, 4));
	}
	else set_Leph_from_jdftx_data();

	if (coul_model != nullptr && clp.update) coul_model->init(this->f_eq);

	for (int iD = 0; iD < eip.ni.size(); iD++)
	if (!alg.only_ee){
		if (eimp[iD]->eimp_model != nullptr && eimp[iD]->eimp_model->imsig != nullptr) zeros(eimp[iD]->eimp_model->imsig, nk_glob, nb);
		add_Lsc_contrib("eimp");
	}
	if (ee_model != nullptr && !alg.only_eimp){
		if (ee_model->imsig != nullptr) zeros(ee_model->imsig, nk_glob, nb);
		ee_model->f = this->f_eq;
		add_Lsc_contrib("ee"); 
	}

	mp->allreduce(Lscii, nk_glob, (int)std::pow(nb, 4), MPI_SUM);
	// add parts "-\sum_3 [ (1-f3) conj(P_331a) delta_2b + delta_1a P_b233 f3 ] where k1=k2=ka=kb"
	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];
		if (ik_glob == ikp_glob){
			axbyc(Lscij[ikpair_local], Lscii[ik_glob], (int)std::pow(nb, 4), c1, c1);
			axbyc(Lscji[ikpair_local], Lscii[ik_glob], (int)std::pow(nb, 4), c1, c1);
		}
	}
	//set_sparseP(false);
}

void electronphonon::add_Lsc_contrib(string what, int iD){
	/*
	ostringstream convert; convert << mp->myrank;
	convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
	FILE *fp; string fname = dir_debug + "debug_add_P" + what + ".out." + convert.str();
	if (exists(fname)) fname = dir_debug + "debug_add_P" + what + "_updated.out." + convert.str();
	bool ldebug = DEBUG;
	if (ldebug) fp = fopen(fname.c_str(), "a");
	MPI_Barrier(MPI_COMM_WORLD);
	*/
	if (ionode && what == "eimp") printf("\nAdd electron-impurity (%d) scattering contribution to Lsc\n", iD);
	if (ionode && what == "ee") printf("\nAdd electron-electron scattering contribution to Lsc\n");
	complex *P1add = new complex[(int)std::pow(nb, 4)]; complex *P2add = new complex[(int)std::pow(nb, 4)];
	complex *P3add = new complex[(int)std::pow(nb, 4)]; complex *P4add = new complex[(int)std::pow(nb, 4)];
	complex *P5add = new complex[(int)std::pow(nb, 4)]; complex *P6add = new complex[(int)std::pow(nb, 4)];

	double nk3_accum = 0;
	double scale_fac = scale_scatt;
	if (what == "eimp") scale_fac = scale_ei;
	if (what == "ee") scale_fac = scale_ee;
	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];
		//if (ldebug) { fprintf(fp, "\nikpair=%d(%d) k1=%d k2=%d\n", ikpair_local, nkpair_proc, ik_glob, ikp_glob); fflush(fp); }
		zeros(P1add, (int)std::pow(nb, 4)); zeros(P2add, (int)std::pow(nb, 4));

		if (what == "eimp" && eip.impMode[iD] == "model_ionized") eimp[iD]->eimp_model->calc_P(ik_glob, ikp_glob, P1add, P2add, false);
		if (what == "eimp" && eip.impMode[iD] == "ab_neutral") eimp[iD]->read_ldbd_imp_P(ikpair_local, P1add, P2add);
		if (what == "ee"){
			nk3_accum += ee_model->calc_P(ik_glob, ikp_glob, P1add, P2add, false);
			if (eep.eeMode != "Pee_fixed_at_eq"){
				if (ionode && DEBUG && !eep.antisymmetry) printf("Compute P3(4) matrices\n");
				if (!eep.antisymmetry) ee_model->calc_P3P4(ik_glob, ikp_glob, P3add, P4add);
				if (ionode && DEBUG && !eep.antisymmetry) printf("Compute P5(6) matrices\n");
				ee_model->calc_P5P6(ik_glob, ikp_glob, P5add, P6add);
			}
		}

		if (scale_fac != 1.) axbyc(P1add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));
		if (scale_fac != 1.) axbyc(P2add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));
		add_Lscii_from_P1_P2(P1add, P2add, ik_glob, ikp_glob);
		add_Lscij_from_P1_P2(P1add, P2add, ikpair_local, ik_glob, ikp_glob, eep.antisymmetry);
		if (what == "ee" && eep.eeMode != "Pee_fixed_at_eq"){
			if (!eep.antisymmetry){
				if (scale_fac != 1.) axbyc(P3add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));
				if (scale_fac != 1.) axbyc(P4add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));
				add_Lscij_from_P3_P4(P3add, P4add, ikpair_local, ik_glob, ikp_glob);
			}
			if (scale_fac != 1.) axbyc(P5add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));
			if (scale_fac != 1.) axbyc(P6add, nullptr, (int)std::pow(nb, 4), c0, complex(scale_fac, 0));
			add_Lscij_from_P5_P6(P5add, P6add, ikpair_local, ik_glob, ikp_glob);
		}
		if (ionode && what == "ee" && ikpair_local % 1000 == 0)  printf("kpair %d done\n", ikpair_local);
		//if (ldebug){
		//	fprintf_complex_mat(fp, P1add, nb*nb, "P1add:"); fflush(fp);
		//	fprintf_complex_mat(fp, P2add, nb*nb, "P2add:"); fflush(fp);
		//}
	}
	delete[] P1add; delete[] P2add; delete[] P3add; delete[] P4add; delete[] P5add; delete[] P6add;
	double max = nk3_accum, min = nk3_accum, avg = nk3_accum;
	mp->allreduce(max, MPI_MAX); mp->allreduce(min, MPI_MIN); mp->allreduce(avg, MPI_SUM); avg /= mp->nprocs;
	if (what == "ee" && ionode) printf("nk3_accum: max= %lg min= %lg avg = %lg\n", max, min, avg);

	if (what == "eimp" && eimp[iD]->eimp_model != nullptr) eimp[iD]->eimp_model->reduce_imsig(this->mp);
	if (what == "ee") ee_model->reduce_imsig(this->mp);
	//if (ldebug) fclose(fp);
}

void electronphonon::set_Leph_from_jdftx_data(){
	if (!alg.Pin_is_sparse){
		if (ionode) { printf("\nread P? matrices and linearize them\n"); fflush(stdout); }
		string fname1 = "ldbd_data/ldbd_P1_", fname2 = "ldbd_data/ldbd_P2_", suffix;
		suffix = isHole ? alg.scatt + "_hole" : alg.scatt;
		fname1 += (suffix + ".bin"); fname2 += (suffix + ".bin");

		size_t Psize = (size_t)std::pow(nb, 4);
		size_t expected_size = nkpair_glob*Psize * 2 * sizeof(double);

		FILE *fp1 = fopen(fname1.c_str(), "rb");
		check_file_size(fp1, expected_size, fname1 + " size does not match expected size");
		fseek_bigfile(fp1, ikpair0_glob, Psize * 2 * sizeof(double));
		FILE *fp2 = fopen(fname2.c_str(), "rb");
		check_file_size(fp2, expected_size, fname1 + " size does not match expected size");
		fseek_bigfile(fp2, ikpair0_glob, Psize * 2 * sizeof(double));

		MPI_Barrier(MPI_COMM_WORLD);
		complex *P1add = new complex[(int)std::pow(nb, 4)]; complex *P2add = new complex[(int)std::pow(nb, 4)];
		for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
			int ik_glob = k1st[ikpair_local];
			int ikp_glob = k2nd[ikpair_local];
			fread(P1add, 2 * sizeof(double), Psize, fp1);
			fread(P2add, 2 * sizeof(double), Psize, fp2);
			if (scale_eph != 1.) axbyc(P1add, nullptr, Psize, c0, complex(scale_eph, 0));
			if (scale_eph != 1.) axbyc(P2add, nullptr, Psize, c0, complex(scale_eph, 0));
			add_Lscii_from_P1_P2(P1add, P2add, ik_glob, ikp_glob);
			add_Lscij_from_P1_P2(P1add, P2add, ikpair_local, ik_glob, ikp_glob);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		delete[] P1add; delete[] P2add;

		fclose(fp1);
		fclose(fp2);
	}
	else{
		error_message("linearization for sparse P matrices is not implemented");
	}
}

void electronphonon::add_Lscii_from_P1_P2(complex *P1_ikjk, complex *P2_ikjk, int ik, int jk){
	for (int i1 = 0; i1 < nb; i1++)
	for (int i2 = 0; i2 < nb; i2++){
		int i12 = i1*nb + i2;
		int n12 = i12*nb*nb;
		// ddmdt^ik_12 = -delta^ik_{1a} P1^{ikjk}_{b2,33} f^jk_3 ddm^ik_ab
		// ddmdt^jk_12 = -delta^jk_{1a} P1^{jkik}_{b2,33} f^ik_3 ddm^jk_ab
		//             = -delta^jk_{1a} conj(P2^{ikjk}_{b2,33}) f^ik_3 ddm^jk_ab
		for (int ib = 0; ib < nb; ib++){
			int nb2 = (ib*nb + i2)*nb*nb;
			int iab = i1*nb + ib; // i1==ia
			int i12ab = n12 + iab;
			for (int i3 = 0; i3 < nb; i3++){
				int ib233 = nb2 + i3*nb + i3;
				Lscii[ik][i12ab] -= P1_ikjk[ib233] * f_eq[jk][i3];
				if (jk > ik) Lscii[jk][i12ab] -= conj(P2_ikjk[ib233]) * f_eq[ik][i3];
			}
		}
		// ddmdt^ik_12 = -(1-f^jk_3) P2^{ikjk}_{33,1a} delta^ik_{2b} ddm^ik_ab
		// ddmdt^jk_12 = -(1-f^ik_3) P2^{jkik}_{33,1a} delta^jk_{2b} ddm^jk_ab
		//             = -(1-f^ik_3) conj(P1^{ikjk}_{33,1a}) delta^jk_{2b} ddm^jk_ab
		for (int ia = 0; ia < nb; ia++){
			int i1a = i1*nb + ia;
			int iab = ia*nb + i2; // i2==ib
			int i12ab = n12 + iab;
			for (int i3 = 0; i3 < nb; i3++){
				int i331a = (i3*nb + i3)*nb*nb + i1a;
				Lscii[ik][i12ab] -= (1 - f_eq[jk][i3]) * P2_ikjk[i331a];
				if (jk > ik) Lscii[jk][i12ab] -= (1 - f_eq[ik][i3]) * conj(P1_ikjk[i331a]);
			}
		}
	}
}

void electronphonon::add_Lscij_from_P1_P2(complex *P1, complex *P2, int ikpair_local, int ik, int jk, bool fac2){
	for (int i1 = 0; i1 < nb; i1++)
	for (int i2 = 0; i2 < nb; i2++){
		int i12 = i1*nb + i2;
		int n12 = i12*nb*nb;
		for (int ia = 0; ia < nb; ia++)
		for (int ib = 0; ib < nb; ib++){
			int iab = ia*nb + ib;
			int i12ab = n12 + iab;
			int iab12 = iab*nb*nb + i12;
			// ddmdt^ik_12 = 2 (1-f^ik_1) P1^ikjk_{12,ab} ddm^jk_ab
			// ddmdt^ik_12 = 2 P2^ikjk_{ab,12} f^ik_2 ddm^jk_ab
			complex ctmp = ((1 - f_eq[ik][i1]) * P1[i12ab] + P2[iab12] * f_eq[ik][i2]);
			Lscij[ikpair_local][i12ab] += (fac2 ? 2 * ctmp : ctmp);
			// ddmdt^jk_12 = 2 (1-f^jk_1) P1^jkik_{12,ab} ddm^ik_ab 
			//             = 2 (1-f^jk_1) conj(P2^ikjk_{12,ab}) ddm^ik_ab
			// ddmdt^jk_12 = 2 P2^jkik_{ab,12} f^jk_2 ddm^ik_ab
			//             = 2 conj(P1^ikjk_{ab,12}) f^jk_2 ddm^ik_ab
			ctmp = ((1 - f_eq[jk][i1]) * conj(P2[i12ab]) + conj(P1[iab12]) * f_eq[jk][i2]);
			Lscji[ikpair_local][i12ab] += (fac2 ? 2 * ctmp : ctmp);
		}
	}
}

void electronphonon::add_Lscij_from_P3_P4(complex *P3, complex *P4, int ikpair_local, int ik, int jk){
	for (int i1 = 0; i1 < nb; i1++)
	for (int i2 = 0; i2 < nb; i2++){
		int i12 = i1*nb + i2;
		int n12 = i12*nb*nb;
		for (int ia = 0; ia < nb; ia++)
		for (int ib = 0; ib < nb; ib++){
			int iab = ia*nb + ib;
			int i12ab = n12 + iab;
			int iab12 = iab*nb*nb + i12;
			// ddmdt^ik_12 = -(1-f^ik_1) P3^ikjk_{12,ab} ddm^jk_ab
			// ddmdt^ik_12 = -P4^ikjk_{12,ab} f^ik_2 ddm^jk_ab
			Lscij[ikpair_local][i12ab] -= ((1 - f_eq[ik][i1]) * P3[i12ab] + P4[i12ab] * f_eq[ik][i2]);
			// ddmdt^jk_12 = -(1-f^jk_1) P3^jkik_{12,ab} ddm^ik_ab
			//             = -(1-f^jk_1) conj(P4^ikjk_{ab,12}) ddm^ik_ab
			// ddmdt^jk_12 = -P4^jkik_{12,ab} f^jk_2 ddm^ik_ab
			//             = -conj(P3^ikjk_{ab,12}) f^jk_2 ddm^ik_ab
			Lscji[ikpair_local][i12ab] -= ((1 - f_eq[jk][i1]) * conj(P4[iab12]) + conj(P3[iab12]) * f_eq[jk][i2]);
		}
	}
}

void electronphonon::add_Lscij_from_P5_P6(complex *P5, complex *P6, int ikpair_local, int ik, int jk){
	for (int i1 = 0; i1 < nb; i1++)
	for (int i2 = 0; i2 < nb; i2++){
		int n12 = (i1*nb + i2)*nb*nb;
		int i21 = i2*nb + i1;
		for (int ia = 0; ia < nb; ia++)
		for (int ib = 0; ib < nb; ib++){
			int i12ab = n12 + ia*nb + ib;
			int iba21 = (ib*nb + ia)*nb*nb + i21;
			// ddmdt^ik_12 = -(1-f^ik_1) P5^ikjk_{12,ab} ddm^jk_ab
			// ddmdt^ik_12 = -P6^ikjk_{12,ab} f^ik_2 ddm^jk_ab
			Lscij[ikpair_local][i12ab] -= ((1 - f_eq[ik][i1]) * P5[i12ab] + P6[i12ab] * f_eq[ik][i2]);
			// ddmdt^jk_12 = -(1-f^jk_1) P5^jkik_{12,ab} ddm^ik_ab
			//             = -(1-f^jk_1) P5^ikjk_{ba,21} ddm^ik_ab
			// ddmdt^jk_12 = -P6^jkik_{12,ab} f^jk_2 ddm^ik_ab
			//             = -P6^ikjk_{ba,21} f^jk_2 ddm^ik_ab
			Lscji[ikpair_local][i12ab] -= ((1 - f_eq[jk][i1]) * P5[iba21] + P6[iba21] * f_eq[jk][i2]);
		}
	}
}