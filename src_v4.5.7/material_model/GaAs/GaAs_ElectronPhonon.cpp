#include "GaAs_ElectronPhonon.h"

void electronphonon_gaas::set_eph(){
	alloc_ephmat(mp->varstart, mp->varend); // allocate matrix A or P
	set_kpair();
	set_ephmat();
}

void electronphonon_gaas::set_kpair(){
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
void electronphonon_gaas::write_ldbd_kpair(){
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

void electronphonon_gaas::set_ephmat(){
	write_ldbd_eph();
}

inline void electronphonon_gaas::g_model_gaas(double q, int im, double wq, complex vk[2 * 2], complex vkp[2 * 2], complex g[2 * 2]){
	double gq;
	gq = 0;
	g[0] = gq; g[1] = 0; g[2] = 0; g[3] = gq;
	// gkkp = evc_k^dagger * g^ms * evc_kp, where g^ms is g in ms basis
	complex maux[2 * 2];
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nb, nb, nb,
		&c1, g, nb, vkp, nb, &c0, maux, nb);
	cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, nb, nb, nb,
		&c1, vk, nb, maux, nb, &c0, g, nb);
}

void electronphonon_gaas::write_ldbd_eph(){
	write_ldbd_eph("P1");
	write_ldbd_eph("P2");
}
void electronphonon_gaas::write_ldbd_eph(string which){
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