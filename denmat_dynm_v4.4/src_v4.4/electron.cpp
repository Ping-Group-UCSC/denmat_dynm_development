#include "electron.h"

void electron::set_H_BS(int ik0_glob, int ik1_glob){
	int nk_proc = ik1_glob - ik0_glob;
	H_BS = alloc_array(nk_proc, nb_dm*nb_dm);
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		// without orbital term, Hz = gs mu_B (B \cdot S), gs is assumed 2, as mu_B = 0.5, Hz will be (B \cdot S)
		vec3_dot_vec3array(H_BS[ik_local], B, s[ik_glob], nb_dm); //vec3_dot_vec3array(complex *vm, vector3<complex> v, complex **m, int n);
	}
}

void electron::compute_dm_Bpert_1st(vector3<> Bpert, double t0){
	MPI_Barrier(MPI_COMM_WORLD);
	if (ionode) printf("enter compute_dm_Bpert_1st\n");
	MPI_Barrier(MPI_COMM_WORLD);
	degthr = 1e-8;

	dm_Bpert = alloc_array(nk, nb_dm*nb_dm);

	for (int ik = 0; ik < nk; ik++){
		for (int i = 0; i < nb_dm; i++)
		for (int j = 0; j < nb_dm; j++){
			complex H1 = Bpert[0] * s[ik][0][i*nb_dm + j] + Bpert[1] * s[ik][1][i*nb_dm + j] + Bpert[2] * s[ik][2][i*nb_dm + j];
			if (fabs(e_dm[ik][i] - e_dm[ik][j]) < degthr){
				double favg = 0.5 * (f_dm[ik][i] + f_dm[ik][j]);
				double dfde = favg * (favg - 1.) / temperature;
				dm_Bpert[ik][i*nb_dm + i] = f_dm[ik][i] + dfde * H1;
			}
			else{
				double dfde = (f_dm[ik][i] - f_dm[ik][j]) / (e_dm[ik][i] - e_dm[ik][j]);
				dm_Bpert[ik][i*nb_dm + j] = dfde * H1;
				if (alg.picture == "interaction") dm_Bpert[ik][i*nb_dm + j] *= cis((e_dm[ik][i] - e_dm[ik][j])*t0);
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (ionode) printf("compute_dm_Bpert_1st done\n");
	MPI_Barrier(MPI_COMM_WORLD);
}

void electron::read_ldbd_size(){
	if (ionode) printf("\nread ldbd_size.dat:\n");
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		sscanf(s, "%d %d %d %d", &nb, &nv, &bStart_dm, &bEnd_dm); if (ionode) printf("nb= %d nv= %d bStart_dm= %d bEnd_dm= %d\n", nb, nv, bStart_dm, bEnd_dm);
		nc = nb - nv;
		nb_dm = bEnd_dm - bStart_dm; nv_dm = std::max(nv - bStart_dm, 0); nc_dm = std::min(nb_dm - nv_dm, 0);
		if (nv < 0 && nv > nb)
			error_message("0 <= nv <= nb");
		if (bStart_dm < 0 || bStart_dm > bEnd_dm || bEnd_dm > nb)
			error_message("0 <= bStart_dm <= bEnd_dm <= nb");
		if (!alg.eph_sepr_eh && (nv == 0 || nv == nb))
			error_message("if there is only hole or electron, it is suggested to set alg.eph_sepr_eh true", "read_ldbd_size");
		if (alg.eph_need_hole && (nv == 0 || bStart_dm >= nv))
			error_message("alg.eph_need_hole but nv is 0 or bStart_dm >= nv", "read_ldbd_size");
		if (alg.eph_need_elec && (nv == nb || bEnd_dm <= nv))
			error_message("alg.eph_need_elec but nv is nb or bEnd_dm <= nv", "read_ldbd_size");
	}
	if (fgets(s, sizeof s, fp) != NULL){
		sscanf(s, "%d %d", &nk_full, &nk); if (ionode) printf("nk_full = %d nk = %d\n", nk_full, nk);
	}
	fclose(fp);
}
void electron::read_ldbd_kvec(){
	if (ionode) printf("\nread ldbd_kvec.bin:\n");
	FILE *fp = fopen("ldbd_data/ldbd_kvec.bin", "rb");
	size_t expected_size = nk * 3 * sizeof(double);
	check_file_size(fp, expected_size, "ldbd_kvec.bin size does not match expected size");
	vector3<> ktmp;
	for (int ik = 0; ik < nk; ik++){
		fread(&ktmp[0], sizeof(double), 3, fp);
		kvec.push_back(ktmp);
	}
	fclose(fp);
}
void electron::read_ldbd_ek(){
	if (ionode) printf("\nread ldbd_ek.bin:\n");
	FILE *fp = fopen("ldbd_data/ldbd_ek.bin", "rb");
	size_t expected_size = nk*nb*sizeof(double);
	check_file_size(fp, expected_size, "ldbd_ek.bin size does not match expected size");
	for (int ik = 0; ik < nk; ik++){
		fread(e[ik], sizeof(double), nb, fp);
		for (int i = 0; i < nb; i++)
			f[ik][i] = fermi(temperature, mu, e[ik][i]);
	}
	if (ionode){
		for (int ik = 0; ik < std::min(nk, 20); ik++){
			printf("ik= %d e=", ik);
			for (int i = 0; i < nb; i++)
				printf(" %lg", e[ik][i]);
			printf("\n");
		}
	}
	fclose(fp);
}
void electron::read_ldbd_smat(){
	if (ionode) printf("\nread ldbd_smat.bin:\n");
	FILE *fp = fopen("ldbd_data/ldbd_smat.bin", "rb");
	size_t expected_size = nk * 3 * nb_dm*nb_dm * 2 * sizeof(double);
	check_file_size(fp, expected_size, "ldbd_smat.bin size does not match expected size");
	for (int ik = 0; ik < nk; ik++)
	for (int idir = 0; idir < 3; idir++)
		fread(s[ik][idir], 2 * sizeof(double), nb_dm*nb_dm, fp);
	if (ionode){
		for (int ik = 0; ik < std::min(nk, 20); ik++)
			printf_complex_mat(s[ik][2], nb_dm, "");
	}
	fclose(fp);
}
void electron::read_ldbd_vmat(){
	if (ionode) printf("\nread ldbd_vmat.bin:\n");
	FILE *fp = fopen("ldbd_data/ldbd_vmat.bin", "rb");
	size_t expected_size = nk * 3 * nb_dm*nb * 2 * sizeof(double);
	check_file_size(fp, expected_size, "ldbd_vmat.bin size does not match expected size");
	for (int ik = 0; ik < nk; ik++)
	for (int idir = 0; idir < 3; idir++)
		fread(v[ik][idir], 2 * sizeof(double), nb_dm*nb, fp);
	//if (ionode){
	//	for (int ik = 0; ik < std::min(nk, 20); ik++)
	//		printf_complex_mat(v[ik][2], nb_dm, nb, "");
	//}
	fclose(fp);
}

void electron::alloc_mat(){
	e = alloc_real_array(nk, nb);
	f = alloc_real_array(nk, nb);
	s = alloc_array(nk, 3, nb_dm*nb_dm);
	v = alloc_array(nk, 3, nb_dm*nb);
}

vector3<> electron::get_kvec(int& ik1, int& ik2, int& ik3){
	vector3<> result;
	result[0] = ik1 / (double)nk1; result[1] = ik2 / (double)nk2; result[2] = ik3 / (double)nk3;
	return result;
}