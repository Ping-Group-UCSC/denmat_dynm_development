#include "DenMat.h"
#include "electron.h"
#include "mymp.h"

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

void singdenmat_k::init_Hcoh(complex **H_BS, double **e){
	this->Hcoh = H_BS;
	Hcoht = new complex[nb*nb]; zeros(Hcoht, nb*nb);
	this->e = e;
}
void singdenmat_k::evolve_coh(double t, complex** ddmdt_coh){
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		compute_Hcoht(t, Hcoh[ik_local], e[ik_glob]);

		// H * dm - dm * H
		zhemm_interface(ddmdt_coh[ik_glob], false, Hcoht, dm[ik_glob], nb);
		zhemm_interface(ddmdt_coh[ik_glob], true, Hcoht, dm[ik_glob], nb, c1, cm1);

		for (int i = 0; i < nb*nb; i++)
			ddmdt_coh[ik_glob][i] *= cmi;
	}

	mp->allreduce(ddmdt_coh, nk_glob, nb*nb, MPI_SUM);
}
void singdenmat_k::compute_Hcoht(double t, complex *Hk, double *ek){
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++)
		Hcoht[i*nb + j] = Hk[i*nb + j] * cis((ek[i] - ek[j])*t);
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
	for (int ik = 0; ik < nk_glob; ik++){
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			oneminusdm[ik][i*nb + j] = -dm[ik][i*nb + j];
		for (int i = 0; i < nb; i++)
			oneminusdm[ik][i*nb + i] = c1 - dm[ik][i*nb + i];
	}
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

void singdenmat_k::set_dm_eq(double t, double **e, int nv){
	if (nb > nv)
		set_dm_eq(false, t, mue, e, nv, nb);
	if (nv > 0)
		set_dm_eq(true, t, muh, e, 0, nv);
}
void singdenmat_k::set_dm_eq(bool isHole, double t, double mu0, double **e, int bStart, int bEnd){
	double ncarrier = 0.;
	for (int ik = 0; ik < nk_glob; ik++)
	for (int i = bStart; i < bEnd; i++)
	if (!isHole)
		ncarrier += real(dm[ik][i*nb + i]);
	else
		ncarrier += (real(dm[ik][i*nb + i]) - 1.); // hole concentration is negative

	zeros(f_eq, nk_glob, nb);
	double mu = find_mu(isHole, ncarrier, t, mu0, e, bStart, bEnd);

	mp->allreduce(f_eq, nk_glob, nb, MPI_SUM);
	for (int ik = 0; ik < nk_glob; ik++)
	for (int i = bStart; i < bEnd; i++)
		dm_eq[ik][i*nb + i] = f_eq[ik][i];

	if (!isHole)
		mue = mu;
	else
		muh = mu;
}
double singdenmat_k::find_mu(bool isHole, double ncarrier, double t, double mu0, double **e, int bStart, int bEnd){
	double result = mu0;
	double damp = 0.7, dmu = 5e-6;
	double excess_old_old = 1., excess_old = 1., excess, ncarrier_new;
	int step = 0;

	while (true){
		ncarrier_new = compute_ncarrier_eq(isHole, t, result, e, bStart, bEnd);
		excess = ncarrier_new - ncarrier;
		if (fabs(excess) > 1e-14){
			if (fabs(excess) > fabs(excess_old) || fabs(excess) > fabs(excess_old_old))
				dmu *= damp;
			result -= sgn(excess) * dmu;

			// the shift of mu should be large when current mu is far from converged one
			if (step > 0 && sgn(excess) == sgn(excess_old)){
				double ratio = ncarrier_new / ncarrier;
				if (ratio < 1e-9)
					result -= sgn(excess) * 10 * t;
				else if (ratio < 1e-4)
					result -= sgn(excess) * 3 * t;
				else if (ratio < 0.1)
					result -= sgn(excess) * 0.7 * t;
			}

			if (dmu < 1e-16){
				ncarrier_new = compute_ncarrier_eq(isHole, t, result, e, bStart, bEnd);
				excess = ncarrier_new - ncarrier;
				break;
			}

			excess_old_old = excess_old;
			excess_old = excess;
			step++;
			if (step > 1e3) break;
		}
		else
			break;
	}

	if (ionode && (fabs(t) < 1e-6 || fabs(excess) > 1e-10)){
		printf("mu0 = %14.7le mu = %14.7le:\n", mu0, result);
		printf("\nCarriers per cell = %lg excess = %lg\n", ncarrier, excess);
	}
	return result;
}
double singdenmat_k::compute_ncarrier_eq(bool isHole, double t, double mu, double **e, int bStart, int bEnd){
	double result = 0.;
	for (int ik = ik0_glob; ik < ik1_glob; ik++)
	for (int i = bStart; i < bEnd; i++){
		f_eq[ik][i] = electron::fermi(t, mu, e[ik][i]);
		if (!isHole)
			result += f_eq[ik][i];
		else
			result += (f_eq[ik][i] - 1.); // hole concentration is negative
	}

	mp->allreduce(result, MPI_SUM);

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
void singdenmat_k::write_ddmdt(){
	MPI_Barrier(MPI_COMM_WORLD);
	if (!ionode) return;
	printf("\nPrint the change of density matrix at this step:\n");
	for (int ik = 0; ik < std::min(nk_glob, 100); ik++){
		for (int i = 0; i < nb*nb; i++)
			printf("(%11.4e,%11.4e) ", ddmdt[ik][i].real(), ddmdt[ik][i].imag());
		printf("\n");
	}
	printf("\n");
}