#include "MoS2_electron.h"

electron_mos2::electron_mos2(parameters *param, lattice_mos2 *latt)
: mp(&mpk), latt(latt), electron(param),
twomeff(0.48 * 2), Omega_z0(5.5e-5), A1(0.055), A2(0.023),
alpha_e(3.3e-4), gs(2.)
{
	set_erange(param->ewind);
	set_brange();
	get_nk();
	write_ldbd_size(param->degauss);

	setState0();
	write_ldbd_kvec();

	setHOmega_mos2();
	write_ldbd_Bso();

	if (alg.modelH0hasBS)
		setState_Bso_mos2();
	write_ldbd_ek();
	write_ldbd_smat();
	write_ldbd_U();
}

void electron_mos2::get_nk(){
	nk = 0;
	for (int ik1 = 0; ik1 < kmesh[0]; ik1++)
	for (int ik2 = 0; ik2 < kmesh[1]; ik2++)
	for (int ik3 = 0; ik3 < kmesh[2]; ik3++){
		vector3<> k = get_kvec(ik1, ik2, ik3);
		double ene = e0k_mos2(k);
		if (ene <= eend && ene >= estart)
			nk++;
	}
}
inline double electron_mos2::e0k_mos2(vector3<>& k){
	return latt->ktoKorKpSq(k) / twomeff;
}

void electron_mos2::setState0(){
	vinfo = new valleyInfo[nk];
	e0 = alloc_real_array(nk, nb);
	alloc_mat(false, true);
	for (int ik = 0; ik < nk; ik++){
		U[ik][0] = c1; U[ik][1] = c0; U[ik][2] = c0; U[ik][3] = c1;
		smat(s[ik]);
	}
	Uh = new complex[nc*nc]{c0}; mtmp = new complex[nc*nc]{c0};

	int ik = 0;
	if (DEBUG && ionode) printf("Print energies:\n");
	for (int ik1 = 0; ik1 < kmesh[0]; ik1++)
	for (int ik2 = 0; ik2 < kmesh[1]; ik2++)
	for (int ik3 = 0; ik3 < kmesh[2]; ik3++){
		vector3<> k = get_kvec(ik1, ik2, ik3);
		double ene = e0k_mos2(k);
		if (ene <= eend && ene >= estart){
			kvec.push_back(k);
			state0_mos2(k, vinfo[ik], e[ik], f[ik]);
			//smat(U[ik], s[ik]); // unnecessary for pure states
			ik++;
		}
	}
	trunc_copy_array(e0, e, nk, 0, nb); // for debug

	if (DEBUG && ionode){
		printf("\nPrint spin matrices:\n");
		for (int ik = 0; ik < nk; ik++){
			printf_complex_mat(s[ik][0], 2, "Sx:");
			printf_complex_mat(s[ik][1], 2, "Sy:");
			printf_complex_mat(s[ik][2], 2, "Sz:");
		}
		printf("\n");
	}
}

void electron_mos2::smat(complex **s){
	s[0][0] = c0;
	s[0][1] = complex(0.5, 0);
	s[0][2] = complex(0.5, 0);
	s[0][3] = c0;
	s[1][0] = c0;
	s[1][1] = 0.5*cmi;
	s[1][2] = 0.5*ci;
	s[1][3] = c0;
	s[2][0] = complex(0.5, 0);
	s[2][1] = c0;
	s[2][2] = c0;
	s[2][3] = complex(-0.5, 0);
}
void electron_mos2::smat(complex *v, complex **s){
	// v^dagger * sigma * v
	hermite(v, Uh, nc);
	for (int idir = 0; idir < 3; idir++){
		zhemm_interface(mtmp, true, s[idir], v, nc);
		zgemm_interface(s[idir], Uh, mtmp, nc);
	}
}

void electron_mos2::state0_mos2(vector3<>& k, valleyInfo& v, double e[2], double f[2]){
	double kVSq;
	v.isK = latt->isKvalley(k, v.kV, kVSq);
	v.k = k;
	e[0] = kVSq / twomeff; e[1] = e[0];
	f[0] = fermi(temperature, mu, e[0]); f[1] = f[0];
	if (DEBUG && ionode)
		printf("isK= %d, kV= %lg %lg %lg, e= %lg, f= %lg\n", v.isK, v.kV[0], v.kV[1], v.kV[2], e[1], f[1]);
}

void electron_mos2::setHOmega_mos2(){
	H_Omega = alloc_array(nk, nb*nb);

	if (DEBUG && ionode) printf("\nPrint H_Omega:\n");
	for (int ik = 0; ik < nk; ik++){
		Bso.push_back(Omega_mos2(vinfo[ik]));
		vec3_dot_vec3array(H_Omega[ik], Bso[ik], s[ik], nc*nc);
		if (DEBUG && ionode){
			printf("ik = %d, kV = (%lg,%lg,%lg), kVlength = %lg\n", ik, vinfo[ik].kV[0], vinfo[ik].kV[1], vinfo[ik].kV[2], latt->klength(vinfo[ik].kV));
			printf_complex_mat(H_Omega[ik], 2, "H_Omega:");
		}
	}
	if (DEBUG && ionode) printf("\n");
}
inline vector3<> electron_mos2::Omega_mos2(valleyInfo& v){
	vector3<> Omega;
	double kVlength = latt->klength(v.kV);
	double sign = v.isK ? 1 : -1;
	vector3<> kV_cart = v.kV * ~latt->Gvec;
	Omega[0] = B[0];
	Omega[1] = B[1];
	Omega[2] = B[2] + sign * (2 * Omega_z0 + A1 * std::pow(kVlength, 2))
		+ A2 * kV_cart[0] * (std::pow(kV_cart[0], 2) - 3 * std::pow(kV_cart[1], 2));
	return Omega;
}

void electron_mos2::setState_Bso_mos2(){
	complex H[nb*nb];
	for (int ik = 0; ik < nk; ik++){
		for (int i = 0; i < nb; i++){
			for (int j = 0; j < nb; j++)
				H[i*nb + j] = H_Omega[ik][i*nb + j];
			H[i*nb + i] += e[ik][i];
		}
		diagonalize(H, nb, e[ik], U[ik]);
		for (int i = 0; i < nb; i++)
			f[ik][i] = fermi(temperature, mu, e[ik][i]);
		smat(U[ik], s[ik]);
	}

	if (DEBUG && ionode){
		printf("\nPrint energies with H0 = Hfree + H_Omega:\n");
		for (int ik = 0; ik < nk; ik++)
			printf("e: %lg %lg, f: %lg %lg\n", e[ik][0], e[ik][1], f[ik][0], f[ik][1]);
		printf("\n");

		printf("\nPrint eigenvectors with H0 = Hfree + H_Omega:\n");
		for (int ik = 0; ik < nk; ik++)
			printf_complex_mat(U[ik], 2, "");
		printf("\n");

		printf("\nPrint spin matrices with H0 = Hfree + H_Omega:\n");
		for (int ik = 0; ik < nk; ik++){
			printf_complex_mat(s[ik][0], 2, "Sx:");
			printf_complex_mat(s[ik][1], 2, "Sy:");
			printf_complex_mat(s[ik][2], 2, "Sz:");
		}
		printf("\n");
	}
}

void electron_mos2::write_ldbd_size(double degauss){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_size.dat", "w");
		fprintf(fp, "Conduction electrons\n");
		fprintf(fp, "%d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph_elec bTop_eph_elec nb_wannier bBot_probe\n",
			nb, nv, nv, nc, nv, nc, nc, nv);
		fprintf(fp, "%21.14le %lu %d %d %d # # nk_full nk kmesh\n", nk_full, nk, kmesh[0], kmesh[1], kmesh[2]);
		fprintf(fp, "%lu # nkpair_elec\n", nk*(nk+1)/2);
		fprintf(fp, "%d %d # modeStart modeStp\n", 0, 9); // does not matter 
		fprintf(fp, "%21.14le # T\n", temperature);
		fprintf(fp, "%21.14le %21.14le %21.14le # muMin, muMax mu (given carrier density)\n", mu, mu, mu);
		fprintf(fp, "%21.14le %21.14le # degauss\n", degauss, 1000.); // does not matter 
		fprintf(fp, "%14.7le %14.7le %14.7le %14.7le %14.7le %14.7le # EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph", estart, eend, estart, eend, estart, eend);
		fclose(fp);
	}
}
void electron_mos2::write_ldbd_kvec(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_kvec.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(&vinfo[ik].k[0], sizeof(double), 3, fp);
		fclose(fp);
	}
}
void electron_mos2::write_ldbd_ek(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_ek.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(e[ik], sizeof(double), nb, fp);
		fclose(fp);
	}
}
void electron_mos2::write_ldbd_smat(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_smat.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
		for (int idir = 0; idir < 3; idir++)
			fwrite(s[ik][idir], 2 * sizeof(double), nb*nb, fp);
		fclose(fp);
	}
}
void electron_mos2::write_ldbd_U(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_Umat.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(U[ik], 2 * sizeof(double), nb*nb, fp);
		fclose(fp);
	}
}
void electron_mos2::write_ldbd_Bso(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_Bso.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(&Bso[ik][0], sizeof(double), 3, fp);
		fclose(fp);
	}
}