#include "GaAs_electron.h"

electron_gaas::electron_gaas(parameters *param, lattice_gaas *latt)
: mp(&mpk), latt(latt), electron(param),
twomeff(0.067 * 2), twogammaD(11.85) // from PRB 79, 125206 (2009)
{
	set_erange(param->ewind);
	set_brange();
	get_nk();
	write_ldbd_size(param->degauss);

	setState0();
	write_ldbd_kvec();

	setHOmega_gaas();
	write_ldbd_Bso();

	if (alg.modelH0hasBS)
		setState_Bso_gaas();
	write_ldbd_ek();
	write_ldbd_smat();
	write_ldbd_U();
}

void electron_gaas::get_nk(){
	nk = 0;
	for (int ik1 = 0; ik1 < kmesh[0]; ik1++)
	for (int ik2 = 0; ik2 < kmesh[1]; ik2++)
	for (int ik3 = 0; ik3 < kmesh[2]; ik3++){
		vector3<> k = get_kvec(ik1, ik2, ik3);
		double ene = e0k_gaas(k);
		if (ene <= eend && ene >= estart)
			nk++;
	}
}
inline double electron_gaas::e0k_gaas(vector3<>& k){
	return latt->klengthSq(k) / twomeff;
}

void electron_gaas::setState0(){
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
		double ene = e0k_gaas(k);
		if (ene <= eend && ene >= estart){
			kvec.push_back(k);
			e[ik][0] = ene; e[ik][1] = ene;
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

void electron_gaas::smat(complex **s){
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
void electron_gaas::smat(complex *v, complex **s){
	// v^dagger * sigma * v
	hermite(v, Uh, nc);
	for (int idir = 0; idir < 3; idir++){
		zhemm_interface(mtmp, true, s[idir], v, nc);
		zgemm_interface(s[idir], Uh, mtmp, nc);
	}
}

void electron_gaas::setHOmega_gaas(){
	H_Omega = alloc_array(nk, nb*nb);

	if (DEBUG && ionode) printf("\nPrint H_Omega:\n");
	for (int ik = 0; ik < nk; ik++){
		Bso.push_back(Omega_gaas(kvec[ik]));
		vec3_dot_vec3array(H_Omega[ik], Bso[ik], s[ik], nc*nc);
		if (DEBUG && ionode){
			printf("ik = %d, k = (%lg,%lg,%lg)\n", ik, kvec[ik][0], kvec[ik][1], kvec[ik][2]);
			printf_complex_mat(H_Omega[ik], 2, "H_Omega:");
		}
	}
	if (DEBUG && ionode) printf("\n");
}
inline vector3<> electron_gaas::Omega_gaas(vector3<>& k){
	vector3<> Omega;
	vector3<> k_cart = wrap(k) * ~latt->Gvec;
	Omega[0] = B[0] + twogammaD * k_cart[0] * (k_cart[1] * k_cart[1] - k_cart[2] * k_cart[2]);
	Omega[1] = B[1] + twogammaD * k_cart[1] * (k_cart[2] * k_cart[2] - k_cart[0] * k_cart[0]);
	Omega[2] = B[2] + twogammaD * k_cart[2] * (k_cart[0] * k_cart[0] - k_cart[1] * k_cart[1]);
	return Omega;
}

void electron_gaas::setState_Bso_gaas(){
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

void electron_gaas::write_ldbd_size(double degauss){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_size.dat", "w");
		fprintf(fp, "Conduction electrons\n");
		fprintf(fp, "%d %d %d %d %d %d %d %d # nb nv bBot_dm bTop_dm bBot_eph_elec bTop_eph_elec nb_wannier bBot_probe\n",
			nb, nv, nv, nc, nv, nc, nc, nv);
		fprintf(fp, "%21.14le %lu %d %d %d # # nk_full nk kmesh\n", nk_full, nk, kmesh[0], kmesh[1], kmesh[2]);
		fprintf(fp, "%lu # nkpair_elec\n", nk*(nk+1)/2);
		fprintf(fp, "%d %d # modeStart modeStp\n", 0, 6); // does not matter 
		fprintf(fp, "%21.14le # T\n", temperature);
		fprintf(fp, "%21.14le %21.14le %21.14le # muMin, muMax mu (given carrier density)\n", mu, mu, mu);
		fprintf(fp, "%21.14le %21.14le # degauss\n", degauss, 1000.); // does not matter 
		fprintf(fp, "%14.7le %14.7le %14.7le %14.7le %14.7le %14.7le # EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph", estart, eend, estart, eend, estart, eend);
		fclose(fp);
	}
}
void electron_gaas::write_ldbd_kvec(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_kvec.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(&kvec[ik][0], sizeof(double), 3, fp);
		fclose(fp);
	}
}
void electron_gaas::write_ldbd_ek(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_ek.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(e[ik], sizeof(double), nb, fp);
		fclose(fp);
	}
}
void electron_gaas::write_ldbd_smat(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_smat.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
		for (int idir = 0; idir < 3; idir++)
			fwrite(s[ik][idir], 2 * sizeof(double), nb*nb, fp);
		fclose(fp);
	}
}
void electron_gaas::write_ldbd_U(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_Umat.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(U[ik], 2 * sizeof(double), nb*nb, fp);
		fclose(fp);
	}
}
void electron_gaas::write_ldbd_Bso(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_Bso.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(&Bso[ik][0], sizeof(double), 3, fp);
		fclose(fp);
	}
}