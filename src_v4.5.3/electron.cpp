#include "electron.h"

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

void electron::set_H_BS(int ik0_glob, int ik1_glob){
	int nk_proc = ik1_glob - ik0_glob;
	H_BS = alloc_array(nk_proc, nb_dm*nb_dm);
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		vector3<> Btot(B);
		if (alg.read_Bso) Btot = B + Bso[ik_glob];
		// without orbital term, Hz = gs mu_B (B \cdot S), gs is assumed 2, as mu_B = 0.5, Hz will be (B \cdot S)
		vec3_dot_vec3array(H_BS[ik_local], Btot, s[ik_glob], nb_dm*nb_dm); //vec3_dot_vec3array(complex *vm, vector3<complex> v, complex **m, int n);
	}
}

void electron::compute_dm_Bpert_1st(vector3<> Bpert, double t0){
	if (ionode) printf("\nenter compute_dm_Bpert_1st\n");
	degthr = 1e-8;

	dm_Bpert = alloc_array(nk, nb_dm*nb_dm);
	trace_sq_ddm_tot = 0;

	for (int ik = 0; ik < nk; ik++){
		// compute ddm_Bpert
		for (int i = 0; i < nb_dm; i++)
		for (int j = i; j < nb_dm; j++){
			complex H1 = Bpert[0] * s[ik][0][i*nb_dm + j] + Bpert[1] * s[ik][1][i*nb_dm + j] + Bpert[2] * s[ik][2][i*nb_dm + j];
			double dfde;
			if (fabs(e_dm[ik][i] - e_dm[ik][j]) < degthr){
				double favg = 0.5 * (f_dm[ik][i] + f_dm[ik][j]);
				dfde = favg * (favg - 1.) / temperature;
			}
			else
				dfde = (f_dm[ik][i] - f_dm[ik][j]) / (e_dm[ik][i] - e_dm[ik][j]);
			dm_Bpert[ik][i*nb_dm + j] = dfde * H1;
			if (i != j && alg.picture == "interaction") dm_Bpert[ik][i*nb_dm + j] *= cis((e_dm[ik][i] - e_dm[ik][j])*t0);
			if (i < j) dm_Bpert[ik][j*nb_dm + i] = dm_Bpert[ik][i*nb_dm + j].conj();
			
			// compute Tr[ddm ddm]
			double norm = dm_Bpert[ik][i*nb_dm + j].norm();
			trace_sq_ddm_tot += (i == j) ? norm : 2 * norm;
		}

		// dm_Bpert = f + ddm_Bpert
		for (int i = 0; i < nb_dm; i++)
			dm_Bpert[ik][i*nb_dm + i] += f_dm[ik][i];
	}
	if (ionode) printf("Tr[ddm_Bpert ddm_Bpert] = %lg\n", trace_sq_ddm_tot / nk);
}

void electron::read_ldbd_size(){
	if (ionode) printf("\nread ldbd_size.dat:\n");
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		if (!alg.eph_need_elec){
			double dtmp1, dtmp2;
			sscanf(s, "%d %d %d %d %d %d %d %d %d %d", &nb, &nv, &bStart_dm, &bEnd_dm, &dtmp1, &dtmp2, &bStart_eph, &bEnd_eph, &nb_wannier, &bskipped_wannier);
		}
		else
			sscanf(s, "%d %d %d %d %d %d %d %d", &nb, &nv, &bStart_dm, &bEnd_dm, &bStart_eph, &bEnd_eph, &nb_wannier, &bskipped_wannier);
		if (ionode) printf("nb= %d nv= %d bStart_dm= %d bEnd_dm= %d bStart_eph = %d bEnd_eph = %d nb_wannier = %d bskipped_wannier = %d\n", nb, nv, bStart_dm, bEnd_dm, bStart_eph, bEnd_eph, nb_wannier, bskipped_wannier);
		nc = nb - nv;
		nb_dm = bEnd_dm - bStart_dm; nv_dm = std::max(nv - bStart_dm, 0); nc_dm = std::min(nb_dm - nv_dm, 0);
		nb_eph = bEnd_eph - bStart_eph; nv_eph = std::max(nv - bStart_eph, 0); nc_eph = std::min(nb_eph - nv_eph, 0);
		if (nv < 0 && nv > nb)
			error_message("0 <= nv <= nb");
		if (bStart_dm < 0 || bStart_dm > bEnd_dm || bEnd_dm > nb)
			error_message("0 <= bStart_dm <= bEnd_dm <= nb");
		if (bStart_eph < bStart_dm || bStart_eph > bEnd_eph || bEnd_eph > bEnd_dm)
			error_message("bStart_dm <= bStart_eph <= bEnd_eph <= bEnd_dm");
		if (!alg.eph_sepr_eh && (nv == 0 || nv == nb))
			error_message("if there is only hole or electron, it is suggested to set alg.eph_sepr_eh true", "read_ldbd_size");
		if (alg.eph_need_hole && (nv == 0 || bStart_dm >= nv || bStart_eph >= nv))
			error_message("alg.eph_need_hole but nv is 0 or bStart_dm >= nv", "read_ldbd_size");
		if (alg.eph_need_elec && (nv == nb || bEnd_dm <= nv || bEnd_dm <= nv))
			error_message("alg.eph_need_elec but nv is nb or bEnd_dm <= nv", "read_ldbd_size");
	}
	if (fgets(s, sizeof s, fp) != NULL){
		sscanf(s, "%le %d %d %d %d", &nk_full, &nk, &kmesh[0], &kmesh[1], &kmesh[2]); if (ionode) printf("nk_full = %21.14le nk = %d kmesh=(%d,%d,%d)\n", nk_full, nk, kmesh[0], kmesh[1], kmesh[2]);
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
void electron::read_ldbd_Bso(){
	if (ionode) printf("\nread ldbd_Bso.bin:\n");
	FILE *fp = fopen("ldbd_data/ldbd_Bso.bin", "rb");
	size_t expected_size = nk * 3 * sizeof(double);
	check_file_size(fp, expected_size, "ldbd_kvec.bin size does not match expected size");
	vector3<> Btmp;
	for (int ik = 0; ik < nk; ik++){
		fread(&Btmp[0], sizeof(double), 3, fp);
		Bso.push_back(Btmp);
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
	if (ionode) print_array_atk(e, nb, "ek:");
	if (ionode) print_array_atk(f, nb, "fk:");
	fclose(fp);
}
void electron::read_ldbd_imsig_eph(){
	if (!exists("ldbd_data/ldbd_imsig.bin")) return;
	if (ionode) printf("\nread ldbd_imsig.bin:\n");
	FILE *fp = fopen("ldbd_data/ldbd_imsig.bin", "rb");
	size_t expected_size = nk*nb_eph*sizeof(double);
	check_file_size(fp, expected_size, "ldbd_imsig.bin size does not match expected size");
	
	imsig_eph_kn = alloc_real_array(nk, nb_eph);
	imsig_eph_k = new double[nk];
	imsig_eph_avg = 0;
	double sum_dfde = 0;

	for (int ik = 0; ik < nk; ik++){
		fread(imsig_eph_kn[ik], sizeof(double), nb_eph, fp);

		// imSig_k = sum_b imSig_kb * dfde_kb / (sum_b dfde_kb), this is just my definition, not must be right
		imsig_eph_k[ik] = 0;
		double sum_dfde_k = 0;
		for (int i = 0; i < nb_eph; i++){
			double dfde = f[ik][i + bStart_eph] * (f[ik][i + bStart_eph] - 1.);
			sum_dfde_k += dfde;
			double dtmp = dfde * imsig_eph_kn[ik][i];
			imsig_eph_k[ik] += dtmp;
			imsig_eph_avg += dtmp;
		}
		imsig_eph_k[ik] /= sum_dfde_k;

		sum_dfde += sum_dfde_k;
	}
	imsig_eph_avg /= sum_dfde;

	if (ionode) print_array_atk(imsig_eph_kn, nb_eph, "imsig_eph_kn in Ha:\n");
	if (ionode) print_array_atk(imsig_eph_k, "imsig_eph_k in Ha:\n");
	if (ionode) printf("imsig_eph_avg = %lg Ha tau_m_avg = %lg fs\n", imsig_eph_avg, 0.5/imsig_eph_avg/fs);
	fclose(fp);
}
void electron::get_kpath(){
	for (int ipath = 0; ipath < nkpath; ipath++){
		vector3<double> kpath_vec = kpath_end[ipath] - kpath_start[ipath];
		double length_kpath = kpath_vec.length();
		std::vector<int> ik_thispath;
		std::vector<double> klength_thispath;
		for (int ik = 0; ik < nk; ik++){
			bool on_path = false;
			for (int ix = 0; ix > -2; ix--){
				for (int iy = 0; iy > -2; iy--){
					for (int iz = 0; iz > -2; iz--){
						vector3<double> krel = kvec[ik] + vector3<>(ix, iy, iz) - kpath_start[ipath];
						double length = krel.length();
						if (length - length_kpath > 1e-10) continue;
						double vecdot = dot(krel, kpath_vec);
						double length_mult = length * length_kpath;
						double div = length_mult == 0 ? 1 : vecdot / length_mult;
						if (fabs(vecdot - length_mult) < 1e-10 && fabs(div - 1) < 1e-10){
							on_path = true;
							ik_thispath.push_back(ik);
							klength_thispath.push_back(length);
						}
						if (on_path) break;
					}
					if (on_path) break;
				}
				if (on_path) break;
			}
		}
		std::vector<int> ind = sort_indexes(klength_thispath);
		for (int ik = 0; ik < ik_thispath.size(); ik++)
			ik_kpath.push_back(ik_thispath[ind[ik]]);
	}
	if (ionode){
		printf("\nPrint k path:\n"); fflush(stdout);
		for (int ik = 0; ik < ik_kpath.size(); ik++){
			printf("%lg %lg %lg\n", kvec[ik_kpath[ik]][0], kvec[ik_kpath[ik]][1], kvec[ik_kpath[ik]][2]); fflush(stdout);
		}
	}
}
void electron::read_ldbd_smat(){
	if (ionode) printf("\nread ldbd_smat.bin:\n");
	FILE *fp = fopen("ldbd_data/ldbd_smat.bin", "rb");
	size_t expected_size = nk * 3 * nb_dm*nb_dm * 2 * sizeof(double);
	check_file_size(fp, expected_size, "ldbd_smat.bin size does not match expected size");
	for (int ik = 0; ik < nk; ik++)
	for (int idir = 0; idir < 3; idir++){
		fread(s[ik][idir], 2 * sizeof(double), nb_dm*nb_dm, fp);
		if (alg.set_scv_zero){
			for (int i = 0; i < nv_dm; i++)
			for (int j = nv_dm; j < nb_dm; j++){
				s[ik][idir][i*nb_dm + j] = c0;
				s[ik][idir][j*nb_dm + i] = c0;
			}
		}
	}
	if (ionode){
		if (ik_kpath.size() > 0)
			for (int ik = 0; ik < ik_kpath.size(); ik++){
				printf("ik= %d:", ik_kpath[ik]);
				printf_complex_mat(s[ik_kpath[ik]][2], nb_dm, "");
			}
		else
			for (int ik = 0; ik < std::min(nk, 20); ik++){
				printf("ik= %d:", ik);
				printf_complex_mat(s[ik][2], nb_dm, "");
			}
		printf("\n");
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
void electron::read_ldbd_Umat(){
	if (ionode) printf("\nread ldbd_Umat.bin:\n");
	if (ionode) printf("get Ufull[:,%d:%d]\n", bStart_eph + bskipped_wannier, bStart_eph + bskipped_wannier + nb_eph);
	FILE *fp = fopen("ldbd_data/ldbd_Umat.bin", "rb");
	size_t expected_size = nk * nb_wannier * nb_wannier * 2 * sizeof(double);
	check_file_size(fp, expected_size, "ldbd_umat.bin size does not match expected size");
	complex *Ufull = new complex[nb_wannier * nb_wannier];
	for (int ik = 0; ik < nk; ik++){
		fread(Ufull, 2 * sizeof(double), nb_wannier * nb_wannier, fp);
		for (int b1 = 0; b1 < nb_wannier; b1++)
		for (int b2 = 0; b2 < nb_eph; b2++)
			U[ik][b1*nb_eph + b2] = Ufull[b1*nb_wannier + b2 + bStart_eph + bskipped_wannier];
	}
	fclose(fp);
}
void electron::print_array_atk(double *a, string s, double unit){
	printf("%s", s.c_str());
	if (ik_kpath.size())
		for (int ik = 0; ik < ik_kpath.size(); ik++)
			printf("ik= %d: %lg\n", ik_kpath[ik], a[ik_kpath[ik]]/unit);
	else
		for (int ik = 0; ik < std::min(nk, 20); ik++)
			printf("ik= %d: %lg\n", ik, a[ik]/unit);
}
void electron::print_array_atk(bool *a, string s){
	printf("%s", s.c_str());
	if (ik_kpath.size())
	for (int ik = 0; ik < ik_kpath.size(); ik++)
		printf("ik= %d: %d\n", ik_kpath[ik], a[ik_kpath[ik]]);
	else
	for (int ik = 0; ik < std::min(nk, 20); ik++)
		printf("ik= %d: %d\n", ik, a[ik]);
}
void electron::print_array_atk(double **a, int n, string s, double unit){
	printf("%s", s.c_str());
	if (ik_kpath.size())
		for (int ik = 0; ik < ik_kpath.size(); ik++){
			printf("ik= %d:", ik_kpath[ik]);
			for (int i = 0; i < n; i++)
				printf(" %lg", a[ik_kpath[ik]][i]/unit);
			printf("\n");
		}
	else
		for (int ik = 0; ik < std::min(nk, 20); ik++){
			printf("ik= %d:", ik);
			for (int i = 0; i < n; i++)
				printf(" %lg", a[ik][i]/unit);
			printf("\n");
		}
}
void electron::print_mat_atk(complex **m, int n, string s){
	printf("%s", s.c_str());
	if (ik_kpath.size())
		for (int ik = 0; ik < ik_kpath.size(); ik++){
			printf("ik= %d:", ik_kpath[ik]);
			printf_complex_mat(m[ik_kpath[ik]], n, "");
		}
	else
		for (int ik = 0; ik < std::min(nk, 20); ik++){
			printf("ik= %d:", ik);
			printf_complex_mat(m[ik], n, "");
		}
	printf("\n");
}

void electron::alloc_mat(bool alloc_v, bool alloc_U){
	e = alloc_real_array(nk, nb);
	f = alloc_real_array(nk, nb);
	s = alloc_array(nk, 3, nb_dm*nb_dm);
	if (alloc_v) v = alloc_array(nk, 3, nb_dm*nb);
	if (alloc_U) U = alloc_array(nk, nb_wannier * nb_eph);
}

vector3<> electron::get_kvec(int& ik1, int& ik2, int& ik3){
	vector3<> result;
	result[0] = ik1 / (double)kmesh[0]; result[1] = ik2 / (double)kmesh[1]; result[2] = ik3 / (double)kmesh[2];
	return result;
}

double electron::find_mu(double carrier_bvk, double t, double mu0){
	if (carrier_bvk == 0) error_message("carrier_bvk cannot be 0", "electron::find_mu");
	bool isHole = carrier_bvk < 0;
	double result = mu0;
	double damp = 0.7, dmu = 5e-6;
	double excess_old_old = 1., excess_old = 1., excess, carrier_bvk_new;
	int step = 0;

	while (true){
		carrier_bvk_new = compute_nfree(isHole, t, result) + eip.compute_carrier_bvk_of_impurity_level(isHole, t, result);
		excess = carrier_bvk_new - carrier_bvk;
		if (fabs(excess) > 1e-14){
			if (fabs(excess) > fabs(excess_old) || fabs(excess) > fabs(excess_old_old))
				dmu *= damp;
			result -= sgn(excess) * dmu;

			// the shift of mu should be large when current mu is far from converged one
			if (step > 0 && sgn(excess) == sgn(excess_old)){
				double ratio = carrier_bvk_new / carrier_bvk;
				if (ratio < 1e-9)
					result -= sgn(excess) * 10 * t;
				else if (ratio < 1e-4)
					result -= sgn(excess) * 3 * t;
				else if (ratio < 0.1)
					result -= sgn(excess) * 0.7 * t;
			}

			if (dmu < 1e-16){
				carrier_bvk_new = compute_nfree(isHole, t, result) + eip.compute_carrier_bvk_of_impurity_level(isHole, t, result);
				excess = carrier_bvk_new - carrier_bvk;
				break;
			}

			excess_old_old = excess_old;
			excess_old = excess;
			step++;
			if (step > 1e4) break;
		}
		else
			break;
	}

	if (ionode){
		printf("isHole = %d mu0 = %14.7le mu = %14.7le:\n", isHole, mu0, result);
		printf("Carriers per bvk * nk = %lg excess = %lg\n", carrier_bvk, excess);
	}
	return result;
}

double electron::compute_nfree(bool isHole, double t, double mu){
	double bStart, bEnd;
	if (isHole){ bStart = 0; bEnd = nv; }
	else { bStart = nv; bEnd = nb; }
	if (bEnd < bStart) error_message("bEnd must be >= bStart", "compute_nfree");
	double result = 0.;
	for (int ik = mp->varstart; ik < mp->varend; ik++)
	for (int i = bStart; i < bEnd; i++){
		double f = fermi(t, mu, e[ik][i]);
		if (!isHole)
			result += f;
		else
			result += (f - 1.); // hole concentration is negative
	}

	mp->allreduce(result, MPI_SUM);

	return result;
}