#include "GaAs_electron.h"

// Notice that H_BS has nk_proc k points while H_Omega has nk_glob k points
void elec_model_GaAs::set_H_BS(int ik0_glob, int ik1_glob){
	int nk_proc = ik1_glob - ik0_glob;
	H_BS = alloc_array(nk_proc, nb*nb);
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		for (int i = 0; i < nb*nb; i++)
			H_BS[ik_local][i] = H_Omega[ik_glob][i];
	}
}

elec_model_GaAs::elec_model_GaAs(lattice_GaAs *latt, parameters *param)
: latt(latt), electron(param),
twomeff(0.134), gs(0.44),
ethr(std::max(param->ewind, param->ewind + param->mu))
{
	// Internal magnetic field = gamma k1 (k2^2 - k3^2)
	// gamma is different in different papers
	// 1. Phys. Stat. Sol. (b) 222, 523 (2000)
	double meff, mcv, Eg, delta_soc, eta;
	meff = 0.067; mcv = 0.8; Eg = 1.55*eV; delta_soc = 0.341*eV; eta = delta_soc / (Eg + delta_soc);
	gamma = 4. / 3.*meff / mcv / sqrt(2.*meff*meff*meff*Eg) * eta / sqrt(1. - eta / 3.);
	if (ionode) printf("gamma = %lg in Phys. Stat. Sol. (b) 222, 523 (2000)", gamma);
	// 2. Sov. Phys. JETP 57, 680 (1983)
	gamma = 6.14597435;
	if (ionode) printf("gamma = %lg in Sov. Phys. JETP 57, 680 (1983)", gamma);
	// 3. PRB 79, 125206 2009
	gamma = 2 * 23.9 * eV * std::pow(Angstrom, 3);
	if (ionode) printf("gamma = %lg in PRB 79, 125206 2009", gamma);

	ns = 1; nb = 2; nv = 0; nc = 2;
	get_nk();
	write_ldbd_size();

	setState0();
	write_ldbd_kvec();

	setHOmega_gaas();

	if (alg.H0hasBS)
		setState_Bso_gaas();
	write_ldbd_ek();
	write_ldbd_smat();

	compute_dm_Bpert(param->Bpert);
	if (alg.H0hasBS)
		zeros(H_Omega, nk, nb*nb);
	write_ldbd_dm0();
}

void elec_model_GaAs::setState0(){
	kvec = new vector3<>[nk];
	alloc_mat();
	e0 = alloc_real_array(nk, nb);
	evc = new complex*[nk];
	for (int ik = 0; ik < nk; ik++){
		evc[ik] = new complex[2 * nb];
		// As eigenstates of free Hamiltonian of the two-band model are pure states,
		// evc[ik][ib*nb+0] and evc[ik][ib*nb+1] are spin-up and spin-down parts of band ib
		evc[ik][0] = c1;
		evc[ik][1] = c0;
		evc[ik][2] = c0;
		evc[ik][3] = c1;
	}

	int ik = 0;
	if (level_debug_dmdyn == 1 && ionode) printf("Print energies:\n");
	for (int ik1 = 0; ik1 < nk1; ik1++)
	for (int ik2 = 0; ik2 < nk2; ik2++)
	for (int ik3 = 0; ik3 < nk3; ik3++){
		vector3<> k = get_kvec(ik1, ik2, ik3);
		double ene = e0k_gaas(k);
		if (ene <= ethr){
			kvec[ik] = k; // needs confirmation
			state0_gaas(k, e[ik], f[ik]);
			for (int i = 0; i < nb; i++)
				e0[ik][i] = e[ik][i]; // for debug
			smat(evc[ik], s[ik]);
			ik++;
		}
	}

	if (level_debug_dmdyn == 1 && ionode){
		printf("\nPrint spin matrices:\n");
		for (int ik = 0; ik < nk; ik++){
			printf_complex_mat(s[ik][0], 2, "Sx:");
			printf_complex_mat(s[ik][1], 2, "Sy:");
			printf_complex_mat(s[ik][2], 2, "Sz:");
		}
		printf("\n");
	}
}

void elec_model_GaAs::smat(complex *v, complex **s){
	// v^dagger * sigma * v
	s[0][0] = 0.5 * (conj(v[2])*v[0] + conj(v[0])*v[2]);
	s[0][1] = 0.5 * (conj(v[2])*v[1] + conj(v[0])*v[3]);
	s[0][2] = 0.5 * (conj(v[3])*v[0] + conj(v[1])*v[2]);
	s[0][3] = 0.5 * (conj(v[3])*v[1] + conj(v[1])*v[3]);
	s[1][0] = 0.5*ci * (conj(v[2])*v[0] - conj(v[0])*v[2]);
	s[1][1] = 0.5*ci * (conj(v[2])*v[1] - conj(v[0])*v[3]);
	s[1][2] = 0.5*ci * (conj(v[3])*v[0] - conj(v[1])*v[2]);
	s[1][3] = 0.5*ci * (conj(v[3])*v[1] - conj(v[1])*v[3]);
	s[2][0] = 0.5 * (conj(v[0])*v[0] - conj(v[2])*v[2]);
	s[2][1] = 0.5 * (conj(v[0])*v[1] - conj(v[2])*v[3]);
	s[2][2] = 0.5 * (conj(v[1])*v[0] - conj(v[3])*v[2]);
	s[2][3] = 0.5 * (conj(v[1])*v[1] - conj(v[3])*v[3]);
}

void elec_model_GaAs::state0_gaas(vector3<>& k, double e[2], double f[2]){
	e[0] = e0k_gaas(k); e[1] = e[0];
	f[0] = fermi(temperature, mu, e[0]); f[1] = f[0];
	if (level_debug_dmdyn == 1 && ionode)
		printf("k= %lg %lg %lg, e= %lg, f= %lg\n", k[0], k[1], k[2], e[1], f[1]);
}

void elec_model_GaAs::get_nk(){
	nk = 0;
	for (int ik1 = 0; ik1 < nk1; ik1++)
	for (int ik2 = 0; ik2 < nk2; ik2++)
	for (int ik3 = 0; ik3 < nk3; ik3++){
		vector3<> k = get_kvec(ik1, ik2, ik3);
		double ene = e0k_gaas(k);
		if (ene <= ethr)
			nk++;
	}
}
inline double elec_model_GaAs::e0k_gaas(vector3<>& k){
	return latt->GGT.metric_length_squared(k) / twomeff;
}

void elec_model_GaAs::setHOmega_gaas(){
	// suppose eigenstates of free Hamiltonian are pure ones and band 0 is spin-up
	H_Omega = new complex*[nk];
	for (int ik = 0; ik < nk; ik++)
		H_Omega[ik] = new complex[nb * nb];

	if (level_debug_dmdyn == 1 && ionode) printf("\nPrint H_Omega:\n");
	for (int ik = 0; ik < nk; ik++){
		vector3<> Omega = Omega_gaas(kvec[ik]);
		H_Omega[ik][0] = 0.25 * gs * complex(Omega[2], 0);
		H_Omega[ik][3] = 0.25 * gs * complex(-Omega[2], 0);
		H_Omega[ik][1] = 0.25 * gs * complex(Omega[0], -Omega[1]);
		H_Omega[ik][2] = 0.25 * gs * complex(Omega[0], Omega[1]);
		if (level_debug_dmdyn == 1 && ionode){
			printf("ik = %d, k = (%lg,%lg,%lg), klength = %lg\n", ik, kvec[ik][0], kvec[ik][1], kvec[ik][2], latt->klength(kvec[ik]));
			printf_complex_mat(H_Omega[ik], 2, "H_Omega:");
		}
	}
	if (level_debug_dmdyn == 1 && ionode) printf("\n");
}
inline vector3<> elec_model_GaAs::Omega_gaas(vector3<> k){
	vector3<> Omega;
	vector3<> k_cart = k * ~latt->Gvec;
	Omega[0] = B[0] + gamma * k_cart[0] * (k_cart[1] * k_cart[1] - k_cart[2] * k_cart[2]);
	Omega[1] = B[1] + gamma * k_cart[1] * (k_cart[2] * k_cart[2] - k_cart[0] * k_cart[0]);
	Omega[2] = B[2] + gamma * k_cart[2] * (k_cart[0] * k_cart[0] - k_cart[1] * k_cart[1]);
	return Omega;
}

void elec_model_GaAs::setState_Bso_gaas(){
	complex H[nb*nb];
	for (int ik = 0; ik < nk; ik++){
		for (int i = 0; i < nb; i++){
			for (int j = 0; j < nb; j++)
				H[i*nb + j] = H_Omega[ik][i*nb + j];
			H[i*nb + i] += e[ik][i];
		}
		diagonalize(H, nb, e[ik], evc[ik]);
		// As eigenstates of free Hamiltonian of the two-band model are pure states,
		// evc[ik][ib*nb+0] and evc[ik][ib*nb+1] are spin-up and spin-down parts of band ib
		for (int i = 0; i < nb; i++)
			f[ik][i] = fermi(temperature, mu, e[ik][i]);
		smat(evc[ik], s[ik]);
	}

	if (level_debug_dmdyn == 1 && ionode){
		printf("\nPrint energies with H0 = Hfree + H_Omega:\n");
		for (int ik = 0; ik < nk; ik++)
			printf("e: %lg %lg, f: %lg %lg\n", e[ik][0], e[ik][1], f[ik][0], f[ik][1]);
		printf("\n");

		printf("\nPrint eigenvectors with H0 = Hfree + H_Omega:\n");
		for (int ik = 0; ik < nk; ik++)
			printf_complex_mat(evc[ik], 2, "");
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

void elec_model_GaAs::compute_dm_Bpert(vector3<> Bpert){
	//dm_Bpert = new complex*[nk];
	//for (int ik = 0; ik < nk; ik++)
	//	dm_Bpert[ik] = new complex[nb * nb];
	dm_Bpert = alloc_array(nk, nb*nb);
	complex H[nb*nb], evc_pert_bandbasis[nb*nb], evc_pert_msbasis[2 * nb];
	double e_pert[nb];

	if (level_debug_dmdyn == 1 && ionode && alg.H0hasBS) printf("\nPrint e_pert and evc_pert:\n");
	for (int ik = 0; ik < nk; ik++){
		vec3_dot_vec3array(H, Bpert, s[ik], nb*nb);
		for (int i = 0; i < nb; i++)//{
			//	for (int j = 0; j < nb; j++)
			//		H[i*nb + j] = Bpert[2] * s[ik][2][i*nb + j];
			H[i*nb + i] += e[ik][i];
		//}
		diagonalize(H, nb, e_pert, evc_pert_bandbasis);

		if (level_debug_dmdyn == 1 && ionode && alg.H0hasBS){
			cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nb, nb, nb,
				&c1, evc[ik], nb, evc_pert_bandbasis, nb, &c0, evc_pert_msbasis, nb);

			printf("ik= %d\n", ik);
			printf("e: %lg %lg\n", e[ik][0], e[ik][1]);
			printf_complex_mat(evc[ik], 2, "evc:");
			printf_complex_mat(H, 2, "H:");
			printf("e_p: %lg %lg\n", e_pert[0], e_pert[1]);
			printf_complex_mat(evc_pert_bandbasis, 2, "evc_p_band:");
			printf_complex_mat(evc_pert_msbasis, 2, "evc_p_ms:");

			for (int i = 0; i < nb; i++){
				H[0] = e0[ik][0] + H_Omega[ik][0] + 0.5*Bpert[2];
				H[3] = e0[ik][1] + H_Omega[ik][3] - 0.5*Bpert[2];
				H[1] = H_Omega[ik][1] + 0.5 * complex(Bpert[0], -Bpert[1]);
				H[2] = H_Omega[ik][2] + 0.5 * complex(Bpert[0], Bpert[1]);
			}
			diagonalize(H, nb, e_pert, evc_pert_msbasis);

			printf_complex_mat(H, 2, "H:");
			printf("e_p: %lg %lg\n", e_pert[0], e_pert[1]);
			printf_complex_mat(evc_pert_msbasis, 2, "evc_p_ms:");
			printf("\n");
		}

		complex f_Bpert[nb*nb], maux[nb*nb];
		f_Bpert[0] = fermi(temperature, mu, e_pert[0]);
		f_Bpert[3] = fermi(temperature, mu, e_pert[1]);
		f_Bpert[1] = c0;
		f_Bpert[2] = c0;
		// under unperturbated eigenvector evc[][], 
		// dm_pert = evc_pert_bandbasis * f_pert * evc_pert_bandbasis^dagger
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, nb, nb, nb,
			&c1, f_Bpert, nb, evc_pert_bandbasis, nb, &c0, maux, nb);
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nb, nb, nb,
			&c1, evc_pert_bandbasis, nb, maux, nb, &c0, dm_Bpert[ik], nb);
	}
	if (level_debug_dmdyn == 1 && ionode){
		printf("\nPrint dm_Bpert:\n");
		for (int ik = 0; ik < nk; ik++)
			printf_complex_mat(dm_Bpert[ik], 2, "dm_Bpert[" + to_string(ik) + "]:");
		printf("\n");
	}
}

void elec_model_GaAs::write_ldbd_size(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_size.dat", "w");
		fprintf(fp, "Conduction electrons\n");
		fprintf(fp, "%d %d # nb nv\n", nb, 0);
		fprintf(fp, "%lu %lu # nkfull nk\n", nk_full, nk);
		fprintf(fp, "%lu %lu # nkpair nkpairh\n", nk*nk, 0);
		fprintf(fp, "%14.7le # T\n", temperature);
		fprintf(fp, "%14.7le %14.7le # muMin, muMax\n", mu, mu);
		fprintf(fp, "%14.7le # degauss\n", 0.); // does not matter 
		fclose(fp);
	}
}
void elec_model_GaAs::write_ldbd_kvec(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_kvec.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(&kvec[ik][0], sizeof(double), 3, fp);
		fclose(fp);
	}
}
void elec_model_GaAs::write_ldbd_ek(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_ek.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(e[ik], sizeof(double), nb, fp);
		fclose(fp);
	}
}
void elec_model_GaAs::write_ldbd_smat(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_smat.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
		for (int idir = 0; idir < 3; idir++)
			fwrite(s[ik][idir], 2 * sizeof(double), nb*nb, fp);
		fclose(fp);
	}
}
void elec_model_GaAs::write_ldbd_dm0(){
	if (ionode){
		FILE *fp = fopen("ldbd_data/ldbd_dm0.bin", "wb");
		for (int ik = 0; ik < nk; ik++)
			fwrite(dm_Bpert[ik], 2 * sizeof(double), nb*nb, fp);
		fclose(fp);
	}
}