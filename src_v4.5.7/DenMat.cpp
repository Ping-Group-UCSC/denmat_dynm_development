#include "DenMat.h"

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

void singdenmat_k::init_Hcoh(complex **H_BS, complex **H_Ez, double **e){
	this->e = e;
	if (H_BS != nullptr || H_Ez != nullptr){
		if (H_BS != nullptr){
			Hcoh = trunc_alloccopy_arraymat(H_BS, nk_proc, nb, 0, nb);
			axbyc(Hcoh, H_Ez, nk_proc, nb*nb, c1, c1);
		}
		else Hcoh = trunc_alloccopy_arraymat(H_Ez, nk_proc, nb, 0, nb);

		if (alg.picture == "interaction"){
			Hcoht = new complex[nb*nb]; zeros(Hcoht, nb*nb);
		}
		else{
			for (int ik_local = 0; ik_local < nk_proc; ik_local++){
				int ik_glob = ik_local + ik0_glob;
				for (int i = 0; i < nb; i++)
					Hcoh[ik_local][i*nb + i] += e[ik_glob][i];
			}
		}
	}
	else { Hcoh = nullptr; }
}
void singdenmat_k::evolve_coh(double t, complex** ddmdt_coh){
	zeros(ddmdt_coh, nk_glob, nb*nb);
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;

		if (Hcoh !=nullptr){
			if (alg.picture == "interaction") compute_Hcoht(t, Hcoh[ik_local], e[ik_glob]); // needed in interaction picture
			else Hcoht = Hcoh[ik_local];

			// H * dm - dm * H
			zhemm_interface(ddmdt_coh[ik_glob], false, Hcoht, dm[ik_glob], nb);
			zhemm_interface(ddmdt_coh[ik_glob], true, Hcoht, dm[ik_glob], nb, c1, cm1);

			for (int i = 0; i < nb*nb; i++)
				ddmdt_coh[ik_glob][i] *= cmi;
		}
		else if (alg.picture == "schrodinger")
			commutator_mat_diag(ddmdt_coh[ik_glob], e[ik_glob], dm[ik_glob], nb, cmi);
	}

	mp->allreduce(ddmdt_coh, nk_glob, nb*nb, MPI_SUM);
}
void singdenmat_k::compute_Hcoht(double t, complex *Hk, double *ek){
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			Hcoht[i*nb + j] = (i == j) ? Hk[i*nb + j] : Hk[i*nb + j] * cis((ek[i] - ek[j])*t);
}

void singdenmat_k::init_dm(double **f){
	zeros(dm, nk_glob, nb*nb); zeros(oneminusdm, nk_glob, nb*nb);
	for (int ik = 0; ik < nk_glob; ik++)
	for (int i = 0; i < nb; i++){
		dm[ik][i*nb + i] = complex(f[ik][i], 0.0);
		oneminusdm[ik][i*nb + i] = complex(1 - f[ik][i], 0.0);
	}
}
void singdenmat_k::init_dm(complex **dm0){
	if (dm0 == NULL){
		if (code == "jdftx"){
			read_ldbd_dm0();
		}
		else
			error_message("when dm0 is null, code must be jdftx");
	}
	else{
		for (int ik = 0; ik < nk_glob; ik++){
			for (int i = 0; i < nb; i++)
			for (int j = 0; j < nb; j++)
				dm[ik][i*nb + j] = 0.5 * (dm0[ik][i*nb + j] + conj(dm0[ik][j*nb + i]));
		}
	}
	set_oneminusdm();
}
void singdenmat_k::read_ldbd_dm0(){
	if (ionode) printf("\nread ldbd_dm0.bin:");
	FILE *fp;
	if (fp = fopen("ldbd_data/ldbd_dm0.bin", "rb")){
		size_t expected_size = nk_glob*nb*nb * 2 * sizeof(double);
		check_file_size(fp, expected_size, "ldbd_dm0.bin size does not match expected size");
		for (int ik = 0; ik < nk_glob; ik++)
			fread(dm[ik], 2 * sizeof(double), nb*nb, fp);
		fclose(fp);
	}
	else{
		error_message("ldbd_dm0.bin does not exist");
	}
}

void singdenmat_k::set_dm_eq(double temperature, double **e, int nv){
	if (nb > nv)
		set_dm_eq(false, temperature, mue, e, nv, nb);
	if (nv > 0)
		set_dm_eq(true, temperature, muh, e, 0, nv);

	compute_f(temperature, mue, muh, e, nv);
	if (ionode && t == t0){
		printf("\ne: ");
		for (int ik = 0; ik < std::min(nk_glob, 10); ik++){
			printf("ik= %d:", ik);
			for (int i = 0; i < nb; i++)
				printf(" %lg", e[ik][i]);
			printf("\n");
		}
		printf("f: ");
		for (int ik = 0; ik < std::min(nk_glob, 10); ik++){
			printf("ik= %d:", ik);
			for (int i = 0; i < nb; i++)
				printf(" %lg", f_eq[ik][i]);
			printf("\n");
		}
	}

	for (int ik = 0; ik < nk_glob; ik++)
	for (int i = 0; i < nb; i++)
		dm_eq[ik][i*nb + i] = f_eq[ik][i];
}
void singdenmat_k::set_dm_eq(bool isHole, double temperature, double mu0, double **e, int bStart, int bEnd){
	double nfree_bvk = 0.;
	for (int ik = 0; ik < nk_glob; ik++)
	for (int i = bStart; i < bEnd; i++)
	if (!isHole) nfree_bvk += real(dm[ik][i*nb + i]);
	else nfree_bvk += (real(dm[ik][i*nb + i]) - 1.); // hole concentration is negative
	if (elec->nk_morek){
		for (int ik = 0; ik < nk_glob; ik++)
		for (int i = bStart; i < bEnd; i++)
		if (!isHole) nfree_bvk -= elec->f_dm[ik][i];
		else nfree_bvk -= (elec->f_dm[ik][i] - 1.); // hole concentration is negative

		for (int ik = 0; ik < elec->nk_morek; ik++)
		for (int i = bStart; i < bEnd; i++)
		if (!isHole) nfree_bvk += elec->f_dm_morek[ik][i];
		else nfree_bvk += (elec->f_dm_morek[ik][i] - 1.); // hole concentration is negative
	}

	double mu = find_mu(isHole, eip.carrier_bvk_ex(isHole, nfree_bvk), temperature, mu0, e, bStart, bEnd);
	if (DEBUG) mu = mu0;
	eip.calc_ni_ionized(isHole, temperature, mu);

	if (!isHole) mue = mu;
	else muh = mu;
}
double singdenmat_k::find_mu(bool isHole, double carrier_bvk, double temperature, double mu0, double **e, int bStart, int bEnd){
	//the subroutine is not efficient and not standard. better to update it if possible
	double result = mu0;
	double damp = 0.7, dmu = 5e-6;
	double excess_old_old = 1., excess_old = 1., excess, carrier_bvk_new;
	int step = 0;

	while (true){
		carrier_bvk_new = compute_nfree_eq(isHole, temperature, result, e, bStart, bEnd) + eip.compute_carrier_bvk_of_impurity_level(isHole, temperature, result);
		excess = carrier_bvk_new - carrier_bvk;
		if (fabs(excess) > 1e-14){
			if (fabs(excess) > fabs(excess_old) || fabs(excess) > fabs(excess_old_old))
				dmu *= damp;
			result -= sgn(excess) * dmu;

			// the shift of mu should be large when current mu is far from converged one
			if (step > 0 && sgn(excess) == sgn(excess_old)){
				double ratio = carrier_bvk_new / carrier_bvk;
				if (ratio < 1e-9)
					result -= sgn(excess) * 10 * temperature;
				else if (ratio < 1e-4)
					result -= sgn(excess) * 3 * temperature;
				else if (ratio < 0.1)
					result -= sgn(excess) * 0.7 * temperature;
			}

			if (dmu < 1e-16){
				carrier_bvk_new = compute_nfree_eq(isHole, temperature, result, e, bStart, bEnd) + eip.compute_carrier_bvk_of_impurity_level(isHole, temperature, result);
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

	if (ionode && (fabs(t) < 1e-6 || fabs(excess) > 1e-10)){
		printf("isHole = %d mu0 = %14.7le mu = %14.7le:\n", isHole, mu0, result);
		printf("Carriers per cell * nk = %lg excess = %lg\n", carrier_bvk, excess);
	}
	return result;
}
double singdenmat_k::compute_nfree_eq(bool isHole, double temperature, double mu, double **e, int bStart, int bEnd){
	// e must be elec->e_dm
	double result = 0.;
	if (elec->nk_morek > 0){
		for (int ik = elec->mp_morek->varstart; ik < elec->mp_morek->varend; ik++)
		for (int i = bStart; i < bEnd; i++){
			double f = electron::fermi(temperature, mu, elec->e_dm_morek[ik][i]);
			if (!isHole)
				result += f;
			else
				result += (f - 1.); // hole concentration is negative
		}

		elec->mp_morek->allreduce(result, MPI_SUM);
	}
	else{
		for (int ik = ik0_glob; ik < ik1_glob; ik++)
		for (int i = bStart; i < bEnd; i++){
			double f = electron::fermi(temperature, mu, e[ik][i]);
			if (!isHole)
				result += f;
			else
				result += (f - 1.); // hole concentration is negative
		}

		mp->allreduce(result, MPI_SUM);
	}

	if (isHole) nh = result;
	else ne = result;
	return result;
}

void singdenmat_k::update_ddmdt(complex **ddmdt_term){
	MPI_Barrier(MPI_COMM_WORLD);
	for (int ik = 0; ik < nk_glob; ik++)
		for (int i = 0; i < nb*nb; i++)
			ddmdt[ik][i] += ddmdt_term[ik][i];
}
void singdenmat_k::update_dm_euler(double dt){
	MPI_Barrier(MPI_COMM_WORLD);
	for (int ik = 0; ik < nk_glob; ik++){
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++){
			dm[ik][i*nb + j] += 0.5 * dt * (ddmdt[ik][i*nb + j] + conj(ddmdt[ik][j*nb + i]));
			oneminusdm[ik][i*nb + j] = -dm[ik][i*nb + j];
		}
		for (int i = 0; i < nb; i++)
			oneminusdm[ik][i*nb + i] = c1 - dm[ik][i*nb + i];
	}
	zeros(ddmdt, nk_glob, nb*nb);
}
void singdenmat_k::set_oneminusdm(){
	MPI_Barrier(MPI_COMM_WORLD);
	for (int ik = 0; ik < nk_glob; ik++){
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++){
			if (i==j)
				oneminusdm[ik][i*nb + i] = c1 - dm[ik][i*nb + i];
			else
				oneminusdm[ik][i*nb + j] = -dm[ik][i*nb + j];
		}
	}
	zeros(ddmdt, nk_glob, nb*nb);
}
/*
void singdenmat_k::init_dmDP(complex **ddm_eq, complex **ddm_neq){
	if (ddm_eq == nullptr || ddm_neq == nullptr) return;
	this->ddm_eq = ddm_eq;
	this->ddm_neq = ddm_neq;
	trace_sq_ddm_eq = new double[nk_glob];
	trace_sq_ddm_neq = new double[nk_glob];
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		trace_sq_ddm_eq[ik_glob] = trace_square_hermite(ddm_eq[ik_glob], nb);
		trace_sq_ddm_neq[ik_glob] = trace_square_hermite(ddm_neq[ik_glob], nb);
	}
}
void singdenmat_k::use_dmDP(double **f){
	if (ddm_eq == nullptr || ddm_neq == nullptr) return;
	complex *ddm = new complex[nb*nb];

	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		for (int i = 0; i < nb; i++)
		for (int b = 0; b < nb; b++)
		if (i == b)
			ddm[i*nb + b] = dm[ik_glob][i*nb + b] - f[ik_glob][i];
		else
			ddm[i*nb + b] = (alg.picture == "schrodinger") ? dm[ik_glob][i*nb + b] : dm[ik_glob][i*nb + b] * cis((e[ik_glob][b] - e[ik_glob][i])*t);

		double weight_ddm_eq = trace_AB(ddm, ddm_eq[ik_glob], nb) / trace_sq_ddm_eq[ik_glob];
		double weight_ddm_neq = trace_AB(ddm, ddm_neq[ik_glob], nb) / trace_sq_ddm_neq[ik_glob];
		for (int ibb = 0; ibb < nb*nb; ibb++)
			ddm[ibb] =  weight_ddm_eq * ddm_eq[ik_glob][ibb] + weight_ddm_neq * ddm_neq[ik_glob][ibb];

		for (int i = 0; i < nb; i++)
		for (int b = 0; b < nb; b++)
		if (i == b)
			dm[ik_glob][i*nb + b] = f[ik_glob][i] + ddm[i*nb + b];
		else
			dm[ik_glob][i*nb + b] = (alg.picture == "schrodinger") ? ddm[i*nb + b] : ddm[i*nb + b] * cis((e[ik_glob][i] - e[ik_glob][b])*t);
	}
}
*/

void singdenmat_k::read_dm_restart(){
	if (ionode) printf("\nread denmat_restart.bin:\n");
	FILE *fp = fopen("restart/denmat_restart.bin", "rb");
	size_t expected_size = nk_glob*nb*nb * 2 * sizeof(double);
	check_file_size(fp, expected_size, "denmat_restart.bin size does not match expected size");
	for (int ik = 0; ik < nk_glob; ik++)
		fread(dm[ik], 2 * sizeof(double), nb*nb, fp);
	fclose(fp);

	for (int ik = 0; ik < nk_glob; ik++){
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			oneminusdm[ik][i*nb + j] = -dm[ik][i*nb + j];
		for (int i = 0; i < nb; i++)
			oneminusdm[ik][i*nb + i] = c1 - dm[ik][i*nb + i];
	}
}
void singdenmat_k::write_dm_tofile(double t){
	MPI_Barrier(MPI_COMM_WORLD);
	if (!ionode) return;
	//FILE *fil = fopen("denmat.out", "a"); // will be too large after long time
	FILE *filbin = fopen("restart/denmat_restart.bin", "wb");
	FILE *filtime = fopen("restart/time_restart.dat", "w"); fprintf(filtime, "%14.7le", t);
	//fprintf(fil, "Writing density matrix at time %10.3e:\n", t);
	for (int ik = 0; ik < nk_glob; ik++)//{
		//for (int i = 0; i < nb*nb; i++)
			//fprintf(fil, "(%15.7e,%15.7e) ", dm[ik][i].real(), dm[ik][i].imag());
		fwrite(dm[ik], 2 * sizeof(double), nb*nb, filbin);
		//fprintf(fil, "\n");
	//}
	//fflush(fil);
	//fclose(fil); 
	fclose(filbin); fclose(filtime);
}
void singdenmat_k::write_dm(){
	MPI_Barrier(MPI_COMM_WORLD);
	if (!ionode) return;
	printf("\nPrint density matrix at time %lg:\n", this->t);
	for (int ik = 0; ik < std::min(nk_glob, 100); ik++){
		for (int i = 0; i < nb*nb; i++)
			printf("(%11.4e,%11.4e) ", dm[ik][i].real(), dm[ik][i].imag());
		printf("\n");
	}
	printf("\n");
}
void singdenmat_k::write_ddmdt(std::vector<vector3<double>> kvec, std::vector<int> ik_kpath, double **e){
	MPI_Barrier(MPI_COMM_WORLD);
	if (!ionode) return;
	string fname = "ddmdt_along_kpath.out"; FILE *fil = fopen(fname.c_str(), "a");
	fprintf(fil, "\nPrint ddmdt at time %21.14ld:\n", t);
	for (int ik = 0; ik < ik_kpath.size(); ik++)
		fprintf_complex_mat(fil, ddmdt[ik_kpath[ik]], nb, "");
	fclose(fil);
}