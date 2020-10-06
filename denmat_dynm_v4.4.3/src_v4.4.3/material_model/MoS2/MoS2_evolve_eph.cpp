#include "common_headers.h"
#include "MoS2_ElectronPhonon.h"
#include "mymp.h"

void eph_model_MoS2::evolve_debug(double t, complex **dm, complex **dm1, complex **ddmdt_eph){
	string dir = "evolve_eph_debug/";
	if (ionode) system("mkdir evolve_eph_debug");
	MPI_Barrier(MPI_COMM_WORLD);
	string fname[nm], fnameinter[nm], fname2[nm], fname2inter[nm]; FILE *fp[nm], *fpinter[nm], *fp2[nm], *fp2inter[nm];
	for (int im = 0; im < nm; im++){
		fname[im] = dir + "evolve_eph.dat.m" + to_string(im) + "." + to_string(mp->myrank);
		fp[im] = fopen(fname[im].c_str(), "w");
		fnameinter[im] = dir + "evolve_eph_inter.dat.m" + to_string(im) + "." + to_string(mp->myrank);
		fpinter[im] = fopen(fnameinter[im].c_str(), "w");
	}

	// dm1 = 1 - dm;
	zeros(ddmdt_eph, nk_glob, nb*nb);

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];

		zeros(ddmdt_contrib, nb*nb);

		bool intra = not(elec_mos2->vinfo[ik_glob].isK xor elec_mos2->vinfo[ikp_glob].isK);

		for (int im = 0; im < nm; im++){
			FILE *fptmp = intra ? fp[im] : fpinter[im];

			if (alg.scatt == "lindblad"){
				fprintf(fptmp, "ik = %d, ikp = %d\n", ik_glob, ikp_glob);
				fprintf_complex_mat(fptmp, dm1[ik_glob], nb, "1-dm_k:");
				fprintf_complex_mat(fptmp, dm[ikp_glob], nb, "dm_k':");
				fprintf_complex_mat(fptmp, App[ikpair_local][im], nb, "App:");
				fprintf_complex_mat(fptmp, Amm[ikpair_local][im], nb, "Amm:");
				fprintf_complex_mat(fptmp, dm1[ikp_glob], nb, "1-dm_k':");
				fprintf_complex_mat(fptmp, dm[ik_glob], nb, "dm_k:");
				fprintf_complex_mat(fptmp, Apm[ikpair_local][im], nb, "Apm:");
				fprintf_complex_mat(fptmp, Amp[ikpair_local][im], nb, "Amp:");
				// + (1-dm_k) * A+_kk' * dm_k' * A+_kk'^dagger
				term1(fptmp, dm1[ik_glob], App[ikpair_local][im], dm[ikp_glob], App[ikpair_local][im]);
				// + (1-dm_k) * A-_kk' * dm_k' * A-_kk'^dagger
				term1(fptmp, dm1[ik_glob], Amm[ikpair_local][im], dm[ikp_glob], Amm[ikpair_local][im]);
				// - A+_kk' * (1-dm_k') * A+_kk'^dagger * dm_k
				term2(fptmp, Apm[ikpair_local][im], dm1[ikp_glob], Apm[ikpair_local][im], dm[ik_glob]);
				// - A-_kk' * (1-dm_k') * A-_kk'^dagger * dm_k
				term2(fptmp, Amp[ikpair_local][im], dm1[ikp_glob], Amp[ikpair_local][im], dm[ik_glob]);
			}
		}

		// H.C. in Eq. 6 of PRB 92, 235423 (2015) probably means the hermite of matrix instead of the hermite of elements
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			ddmdt_eph[ik_glob][i*nb + j] += (prefac_eph*0.5) * (ddmdt_contrib[i*nb + j] + conj(ddmdt_contrib[j*nb + i]));
	}

	mp->allreduce(ddmdt_eph, nk_glob, nb*nb, MPI_SUM);

	if (alg.ddmeq){
		for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
		for (int i = 0; i < nb*nb; i++)
			ddmdt_eph[ik_glob][i] -= ddm_eq[ik_glob][i];
	}

	for (int im = 0; im < nm; im++){
		fclose(fp[im]); fclose(fpinter[im]);
	}
}

inline void electronphonon::term1(FILE *fp, complex *dm1, complex *a, complex *dm, complex *b){
	// + (1-dm_k) * a+-_kk' * dm_k' * b+-_kk'^dagger
	hermite(b, maux2);
	zhemm_interface(maux1, true, dm, maux2, nb); // maux1 = dm_k' * b_kk'^dagger
	zgemm_interface(maux2, a, maux1, nb); // maux2 = a_kk' * maux1
	zhemm_interface(maux1, true, dm1, maux2, nb); // maux1 = (1-dm_k) * maux2
	fprintf_complex_mat(fp, maux1, nb, "term1:");
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] += maux1[i];
}
inline void electronphonon::term2(FILE *fp, complex *a, complex *dm1, complex *b, complex *dm){
	// - a+_kk' * (1-dm_k') * b+_kk'^dagger * dm_k
	hermite(b, maux2);
	zhemm_interface(maux1, false, dm, maux2, nb); // maux1 = b_kk'^dagger * dm_k
	zhemm_interface(maux2, true, dm1, maux1, nb); // maux2 = (1-dm_k') * maux1
	zgemm_interface(maux1, a, maux2, nb); // maux1 = a_kk' * maux2
	fprintf_complex_mat(fp, maux1, nb, "term2:");
	// notice that this terms is substract
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] -= maux1[i];
}
