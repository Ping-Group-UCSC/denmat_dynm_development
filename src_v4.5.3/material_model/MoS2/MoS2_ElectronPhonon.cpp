#include "MoS2_ElectronPhonon.h"

void electronphonon_mos2::set_eph(){
	alloc_ephmat(mp->varstart, mp->varend); // allocate matrix A or P
	set_kpair();
	set_ephmat();
}

void electronphonon_mos2::set_kpair(){
	/*
	k1st[0] = ikpair0_glob / nk_glob;
	k2nd[0] = ikpair0_glob % nk_glob;
	for (int ikpair_local = 1; ikpair_local < nkpair_proc; ikpair_local++){
		if (k2nd[ikpair_local - 1] == nk_glob - 1){
			k2nd[ikpair_local] = 0;
			k1st[ikpair_local] = k1st[ikpair_local - 1] + 1;
		}
		else{
			k2nd[ikpair_local] = k2nd[ikpair_local - 1] + 1;
			k1st[ikpair_local] = k1st[ikpair_local - 1];
		}
	}
	ik0_glob = k1st[0];
	ik1_glob = k1st[nkpair_proc] + 1;
	nk_proc = ik1_glob - ik0_glob;
	*/
	for (size_t ik1 = 0; ik1 < nk_glob; ik1++)
	for (size_t ik2 = ik1; ik2 < nk_glob; ik2++)
		kpairs.push_back(std::make_pair(ik1, ik2));
	if (ionode) printf("Number of pairs: %lu\n\n", kpairs.size());

	if (ionode) std::random_shuffle(kpairs.begin(), kpairs.end());
	if (ionode) printf("Randomly rearranging kpairs done\n");
	mp->bcast((size_t*)kpairs.data(), nkpair_glob * 2);
	if (ionode) printf("bcast kpairs done\n");

	write_ldbd_kpair();
	read_ldbd_kpair();
}
void electronphonon_mos2::write_ldbd_kpair(){
	if (ionode){
		string fnamek = "ldbd_data/ldbd_kpair_k1st.bin"; string fnamekp = "ldbd_data/ldbd_kpair_k2nd.bin";
		FILE *fpk = fopen(fnamek.c_str(), "wb"), *fpkp = fopen(fnamekp.c_str(), "wb");
		for (size_t ikpair = 0; ikpair < nkpair_glob; ikpair++){
			fwrite(&kpairs[ikpair].first, sizeof(size_t), 1, fpk);
			fwrite(&kpairs[ikpair].second, sizeof(size_t), 1, fpkp);
		}
		fclose(fpk); fclose(fpkp);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void electronphonon_mos2::set_ephmat(){
	ostringstream convert; convert << mp->myrank;
	convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
	FILE *fp; string fname = dir_debug + "debug_set_ephmat.out." + convert.str();
	bool ldebug = DEBUG;
	if (ldebug) fp = fopen(fname.c_str(), "a");

	double qlength, qVlength, wq, nq, sqrtdeltaplus, sqrtdeltaminus, deltaplus, deltaminus;
	// for lindblad, G1^+- = G2^+- = g^+- sqrt(delta(ek - ekp +- wq))
	// for conventional, G1^+- = g^+-, G2^+- = g^+- delta(ek - ekp +- wq)
	complex g[nb * nb], G1p[nb * nb], G1m[nb * nb], G2p[nb * nb], G2m[nb * nb];

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];
		if (ldebug) { fprintf(fp, "\nikpair=%d(%d) k1=%d k2=%d\n", ikpair_local, nkpair_proc, ik_glob, ikp_glob); fflush(fp); }

		bool intra = not(elec_mos2->vinfo[ik_glob].isK xor elec_mos2->vinfo[ikp_glob].isK);
		qlength = latt->klength(elec_mos2->vinfo[ik_glob].k - elec_mos2->vinfo[ikp_glob].k);
		qVlength = latt->klength(elec_mos2->vinfo[ik_glob].kV - elec_mos2->vinfo[ikp_glob].kV);
		if (ldebug) { fprintf(fp, "\nintra=%d |q|=%lg |qV|=%lg\n", intra, qlength, qVlength); fflush(fp); }

		for (int im = 0; im < nm; im++){
			wq = ph_mos2->omega_model_mos2(qlength, im);
			nq = ph_mos2->bose(ph_mos2->temperature, wq);
			if (ldebug) { fprintf(fp, "\nim=%d wq=%lg nq=%lg\n", im, wq, nq); fflush(fp); }
			g_model_mos2(intra, qlength, qVlength, im, wq, elec_mos2->U[ik_glob], elec_mos2->U[ikp_glob], g);
			if (ldebug){ fprintf_complex_mat(fp, g, nb, "g:"); fflush(fp); }

			for (int ib = 0; ib < nb; ib++)
			for (int ibp = 0; ibp < nb; ibp++){
				int ibb = ib*nb + ibp;
				if (alg.scatt == "lindblad"){
					sqrtdeltaplus = sqrt_gauss_exp(elec_mos2->e[ik_glob][ib] - elec_mos2->e[ikp_glob][ibp] + wq);
					sqrtdeltaminus = sqrt_gauss_exp(elec_mos2->e[ik_glob][ib] - elec_mos2->e[ikp_glob][ibp] - wq);
					G1p[ibb] = prefac_sqrtgauss * g[ibb] * sqrtdeltaplus; G2p[ibb] = G1p[ibb];
					G1m[ibb] = prefac_sqrtgauss * g[ibb] * sqrtdeltaminus; G2m[ibb] = G1m[ibb];
				}
				else{
					deltaplus = gauss_exp(elec_mos2->e[ik_glob][ib] - elec_mos2->e[ikp_glob][ibp] + wq);
					deltaminus = gauss_exp(elec_mos2->e[ik_glob][ib] - elec_mos2->e[ikp_glob][ibp] - wq);
					G1p[ibb] = g[ibb]; G2p[ibb] = prefac_gauss * g[ibb] * deltaplus;
					G1m[ibb] = g[ibb]; G2m[ibb] = prefac_gauss * g[ibb] * deltaminus;
				}
			}
			if (ldebug){ 
				fprintf_complex_mat(fp, G2p, nb, "G2p:"); fflush(fp); 
				fprintf_complex_mat(fp, G2m, nb, "G2m:"); fflush(fp);
			}

			if (alg.summode){
				// ddmdt = 2pi/Nq [ (1-dm)^k_n1n3 P1^kk'_n3n2,n4n5 dm^k'_n4n5
				//                  + (1-dm)^k'_n3n4 P2^kk'_n3n4,n1n5 dm^k_n5n2 ] + H.C.
				// P1_n3n2,n4n5 = G^+-_n3n4 * conj(G^+-_n2n5) * nq^+-
				// P2_n3n4,n1n5 = G^-+_n1n3 * conj(G^-+_n5n4) * nq^+-
				for (int i1 = 0; i1 < nb; i1++)
				for (int i2 = 0; i2 < nb; i2++){
					int n12 = (i1*nb + i2)*nb*nb;
					for (int i3 = 0; i3 < nb; i3++){
						int i13 = i1*nb + i3;
						int i31 = i3*nb + i1;
						for (int i4 = 0; i4 < nb; i4++){
							P1[ikpair_local][n12 + i3*nb + i4] += G1p[i13] * conj(G2p[i2*nb + i4]) * (nq + 1)
								+ G1m[i13] * conj(G2m[i2*nb + i4]) * nq;
							P2[ikpair_local][n12 + i3*nb + i4] += G1m[i31] * conj(G2m[i4*nb + i2]) * (nq + 1)
								+ G1p[i31] * conj(G2p[i4*nb + i2]) * nq;
						}
					}
				}
				if (ldebug){
					fprintf_complex_mat(fp, P1[ikpair_local], nb*nb, "P1:"); fflush(fp);
					fprintf_complex_mat(fp, P2[ikpair_local], nb*nb, "P2:"); fflush(fp);
				}
			}
			else if (alg.scatt == "lindblad"){
				for (int ib = 0; ib < nb; ib++)
				for (int ibp = 0; ibp < nb; ibp++){
					int ibb = ib*nb + ibp;
					App[ikpair_local][im][ibb] = G1p[ibb] * sqrt(nq + 1);
					Amm[ikpair_local][im][ibb] = G1m[ibb] * sqrt(nq);
					Apm[ikpair_local][im][ibb] = G1p[ibb] * sqrt(nq);
					Amp[ikpair_local][im][ibb] = G1m[ibb] * sqrt(nq + 1);
				}
			}
		}
	}

	write_ldbd_eph();
	if (ldebug) fclose(fp);
}

inline void electronphonon_mos2::g_model_mos2(bool intra, double q, double qV, int im, double wq, complex vk[2 * 2], complex vkp[2 * 2], complex g[2 * 2]){
	double gq;
	if (im == 0){
		gq = prefac_g * (intra ? xita * sqrt(qV / ph_mos2->cta) : d1tainter * qV / sqrt(wq));
	}
	else if (im == 1){
		gq = prefac_g * (intra ? xila * sqrt(qV / ph_mos2->cla) : d1lainter * qV / sqrt(wq));
	}
	else if (im == 2){
		gq = prefac_g * (intra ? d1tointra * qV / sqrt(wq) : d1tointer * qV / sqrt(wq));
	}
	else if (im == 3){
		gq = gfr * erfc(halfdfr * q) + prefac_g * (intra ? 0 : d0lointer / sqrt(wq));
	}
	g[0] = gq; g[1] = 0; g[2] = 0; g[3] = gq;
	// gkkp = evc_k^dagger * g^ms * evc_kp, where g^ms is g in ms basis
	complex maux[2 * 2];
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nb, nb, nb,
		&c1, g, nb, vkp, nb, &c0, maux, nb);
	cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, nb, nb, nb,
		&c1, vk, nb, maux, nb, &c0, g, nb);
}

void electronphonon_mos2::write_ldbd_eph(){
	write_ldbd_eph("P1");
	write_ldbd_eph("P2");
}
void electronphonon_mos2::write_ldbd_eph(string which){
	MPI_Barrier(MPI_COMM_WORLD);
	string subfix = alg.scatt == "lindblad" ? "lindblad" : "conventional";
	string fname = "ldbd_data/ldbd_" + which + "_" + subfix + ".bin";
	FILE *fp = fopen(fname.c_str(), "wb");
	MPI_Barrier(MPI_COMM_WORLD);
	size_t Psize = (size_t)std::pow(nb, 4);
	fseek_bigfile(fp, ikpair0_glob, (2 * sizeof(double)) * Psize);

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		if (which == "P1") fwrite(&P1[ikpair_local][0], 2 * sizeof(double), Psize, fp);
		else fwrite(&P2[ikpair_local][0], 2 * sizeof(double), Psize, fp);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	fclose(fp);
}