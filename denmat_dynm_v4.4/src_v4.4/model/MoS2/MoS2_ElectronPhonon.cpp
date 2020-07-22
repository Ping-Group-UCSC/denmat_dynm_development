#include "MoS2_ElectronPhonon.h"
#include "mymp.h"

void eph_model_MoS2::set_eph(mymp *mp){
	this->mp = mp;
	alloc_ephmat(mp->varstart, mp->varend); // allocate matrix A or P
	set_kpair();
	set_ephmat();
}

void eph_model_MoS2::set_kpair(){
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

	write_ldbd_kpair();
}
void eph_model_MoS2::write_ldbd_kpair(){
	if(ionode) printf("size of size_t = %lu\n", sizeof(size_t));
	MPI_Barrier(MPI_COMM_WORLD);
	FILE *fpk = fopen("ldbd_data/ldbd_kpair_k1st.bin", "wb");
	MPI_Barrier(MPI_COMM_WORLD);
	FILE *fpkp = fopen("ldbd_data/ldbd_kpair_k2nd.bin", "wb");
	MPI_Barrier(MPI_COMM_WORLD);
	int pos = ikpair0_glob * sizeof(size_t);
	MPI_Barrier(MPI_COMM_WORLD);
	fseek(fpk, pos, SEEK_SET);
	fseek(fpkp, pos, SEEK_SET);
	fwrite(k1st, sizeof(size_t), nkpair_proc, fpk);
	fwrite(k2nd, sizeof(size_t), nkpair_proc, fpkp);
	MPI_Barrier(MPI_COMM_WORLD);
	fclose(fpk); fclose(fpkp);
}

void eph_model_MoS2::set_ephmat(){
	double qlength, qVlength, wq, nq, sqrtdeltaplus, sqrtdeltaminus, deltaplus, deltaminus;
	// for lindblad, G1^+- = G2^+- = g^+- sqrt(delta(ek - ekp +- wq))
	// for conventional, G1^+- = g^+-, G2^+- = g^+- delta(ek - ekp +- wq)
	complex g[nb * nb], G1p[nb * nb], G1m[nb * nb], G2p[nb * nb], G2m[nb * nb];

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];

		bool intra = not(elec_mos2->vinfo[ik_glob].isK xor elec_mos2->vinfo[ikp_glob].isK);
		qlength = latt->klength(elec_mos2->vinfo[ik_glob].k - elec_mos2->vinfo[ikp_glob].k);
		qVlength = latt->klength(elec_mos2->vinfo[ik_glob].kV - elec_mos2->vinfo[ikp_glob].kV);

		for (int im = 0; im < nm; im++){
			wq = ph_mos2->omega_model_MoS2(qlength, im);
			nq = ph_mos2->bose(ph_mos2->temperature, wq);
			g_model_MoS2(intra, qlength, qVlength, im, wq, elec_mos2->evc[ik_glob], elec_mos2->evc[ikp_glob], g);

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
}
void eph_model_MoS2::debug_eph_model_MoS2(){
	string dir = "eph_matrix_elements/";
	if (ionode) system("mkdir eph_matrix_elements");
	MPI_Barrier(MPI_COMM_WORLD);
	string fname[nm], fnameinter[nm], fname2[nm], fname2inter[nm]; FILE *fp[nm], *fpinter[nm], *fp2[nm], *fp2inter[nm];
	for (int im = 0; im < nm; im++){
		fname[im] = dir + "eph.dat.m" + to_string(im) + "." + to_string(mp->myrank);
		fp[im] = fopen(fname[im].c_str(), "w");
		fnameinter[im] = dir + "ephinter.dat.m" + to_string(im) + "." + to_string(mp->myrank);
		fpinter[im] = fopen(fnameinter[im].c_str(), "w");
		fname2[im] = dir + "eph2.dat.m" + to_string(im) + "." + to_string(mp->myrank);
		fp2[im] = fopen(fname2[im].c_str(), "w");
		fname2inter[im] = dir + "eph2inter.dat.m" + to_string(im) + "." + to_string(mp->myrank);
		fp2inter[im] = fopen(fname2inter[im].c_str(), "w");
		printf("myrank = %d fname = %s\n", mp->myrank, fname[im].c_str());
	}

	double qlength, qVlength, wq, nq, sqrtdeltaplus, sqrtdeltaminus;
	complex g[nb * nb], Gp[nb * nb], Gm[nb * nb];

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];

		bool intra = not(elec_mos2->vinfo[ik_glob].isK xor elec_mos2->vinfo[ikp_glob].isK);
		qlength = latt->klength(elec_mos2->vinfo[ik_glob].k - elec_mos2->vinfo[ikp_glob].k);
		qVlength = latt->klength(elec_mos2->vinfo[ik_glob].kV - elec_mos2->vinfo[ikp_glob].kV);

		for (int im = 0; im < nm; im++){
			wq = ph_mos2->omega_model_MoS2(qlength, im);
			nq = ph_mos2->bose(ph_mos2->temperature, wq);
			g_model_MoS2(intra, qlength, qVlength, im, wq, elec_mos2->evc[ik_glob], elec_mos2->evc[ikp_glob], g);

			FILE *fptmp = intra ? fp[im] : fpinter[im];
			FILE *fp2tmp = intra ? fp2[im] : fp2inter[im];
			fprintf(fptmp, "%lg %lg\n", qVlength, real(g[0]));
			fprintf(fp2tmp, "%lg %lg %lg %lg \n", elec_mos2->e[ik_glob][0], elec_mos2->e[ikp_glob][0], wq, nq);

			if (alg.scatt == "lindblad"){
				if (alg.summode){
					for (int ib = 0; ib < nb; ib++)
					for (int ibp = 0; ibp < nb; ibp++){
						int ibb = ib*nb + ibp;
						sqrtdeltaplus = sqrt_gauss_exp(elec_mos2->e[ik_glob][ib] - elec_mos2->e[ikp_glob][ibp] + wq);
						sqrtdeltaminus = sqrt_gauss_exp(elec_mos2->e[ik_glob][ib] - elec_mos2->e[ikp_glob][ibp] - wq);
						Gp[ibb] = prefac_sqrtgauss * g[ibb] * sqrtdeltaplus; Gm[ibb] = prefac_sqrtgauss * g[ibb] * sqrtdeltaminus;
					}
					fprintf(fp2tmp, "ik= %d ikp= %d", ik_glob, ikp_glob);
					for (int i1 = 0; i1 < nb; i1++)
					for (int i2 = 0; i2 < nb; i2++){
						int n12 = (i1*nb + i2)*nb*nb;
						int i3 = i1;
						int i13 = i1*nb + i3;
						int i31 = i3*nb + i1;
						int i4 = i2;
						P1[ikpair_local][n12 + i3*nb + i4] += Gp[i13] * conj(Gp[i2*nb + i4]) * (nq + 1)
							+ Gm[i13] * conj(Gm[i2*nb + i4]) * nq;
						P2[ikpair_local][n12 + i3*nb + i4] += Gm[i31] * conj(Gm[i4*nb + i2]) * (nq + 1)
							+ Gp[i31] * conj(Gp[i4*nb + i2]) * nq;
						fprintf(fp2tmp, "i1= %d i2= %d, %lg %lg\n", i1, i2, real(P1[ikpair_local][n12 + i3*nb + i4]), real(P2[ikpair_local][n12 + i3*nb + i4]));
					}
				}
				else{
					sqrtdeltaplus = sqrt_gauss_exp(elec_mos2->e[ik_glob][0] - elec_mos2->e[ikp_glob][0] + wq);
					sqrtdeltaminus = sqrt_gauss_exp(elec_mos2->e[ik_glob][0] - elec_mos2->e[ikp_glob][0] - wq);
					Gp[0] = g[0] * sqrtdeltaplus; Gm[0] = g[0] * sqrtdeltaminus;
					App[ikpair_local][im][0] = Gp[0] * sqrt(nq + 1);
					Amm[ikpair_local][im][0] = Gm[0] * sqrt(nq);
					Apm[ikpair_local][im][0] = Gp[0] * sqrt(nq);
					Amp[ikpair_local][im][0] = Gm[0] * sqrt(nq + 1);
					fprintf(fp2tmp, "ik= %d ikp= %d, %lg %lg %lg %lg\n", ik_glob, ikp_glob,
						real(App[ikpair_local][im][0]), real(Amm[ikpair_local][im][0]), real(Apm[ikpair_local][im][0]), real(Amp[ikpair_local][im][0]));
				}
			}
		}
	}
	for (int im = 0; im < nm; im++){
		fclose(fp[im]); fclose(fpinter[im]); fclose(fp2[im]); fclose(fp2inter[im]);
	}
}

inline void eph_model_MoS2::g_model_MoS2(bool intra, double q, double qV, int im, double wq, complex vk[2 * 2], complex vkp[2 * 2], complex g[2 * 2]){
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

void eph_model_MoS2::write_ldbd_eph(){
	MPI_Barrier(MPI_COMM_WORLD);
	FILE *fp1;
	if (alg.scatt == "lindblad")
		fp1 = fopen("ldbd_data/ldbd_P1_lindblad.bin", "wb");
	else
		fp1 = fopen("ldbd_data/ldbd_P1_conventional.bin", "wb");
	MPI_Barrier(MPI_COMM_WORLD);
	FILE *fp2;
	if (alg.scatt == "lindblad")
		fp2 = fopen("ldbd_data/ldbd_P2_lindblad.bin", "wb");
	else
		fp2 = fopen("ldbd_data/ldbd_P2_conventional.bin", "wb");
	MPI_Barrier(MPI_COMM_WORLD);
	int pos = ikpair0_glob * (2 * sizeof(double)) * ((int)std::pow(nb, 4));
	MPI_Barrier(MPI_COMM_WORLD);
	fseek(fp1, pos, SEEK_SET);
	fseek(fp2, pos, SEEK_SET);

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		fwrite(&P1[ikpair_local][0], 2 * sizeof(double), (int)std::pow(nb, 4), fp1);
		fwrite(&P2[ikpair_local][0], 2 * sizeof(double), (int)std::pow(nb, 4), fp2);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	fclose(fp1); fclose(fp2);
}