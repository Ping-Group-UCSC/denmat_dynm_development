#include "ElectronPhonon.h"

void electronphonon::set_eph(mymp *mp){
	this->mp = mp;
	alloc_ephmat(mp->varstart, mp->varend); // allocate matrix A or P
	set_kpair();
	set_ephmat();
	set_sparseP();
}

void electronphonon::set_sparseP(){
	sP1 = new sparse2D(mp, P1, nkpair_proc, nb*nb, nb*nb, alg.thr_sparseP);
	sP2 = new sparse2D(mp, P2, nkpair_proc, nb*nb, nb*nb, alg.thr_sparseP);
	if (alg.sparseP){
		sP1->sparse(P1, true); // do_test = true
		sP2->sparse(P2, true);
		sm1_next = new sparse_mat((int)std::pow(nb, 4), true);
		sm2_next = new sparse_mat((int)std::pow(nb, 4), true);
		if (alg.expt){
			smat1_time = new sparse_mat((int)std::pow(nb, 4), true);
			smat2_time = new sparse_mat((int)std::pow(nb, 4), true);
			make_map();
		}
	}
}

void electronphonon::get_brange(bool sepr_eh, bool isHole){
	if (ionode) printf("\nread ldbd_size.dat to get band range:\n");
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		int itmp1, itmp2, itmp3, itmp4, itmp5, itmp6;
		if (!isHole)
			sscanf(s, "%d %d %d %d %d %d", &itmp1, &itmp2, &itmp3, &itmp4, &bStart, &bEnd);
		else
			sscanf(s, "%d %d %d %d %d %d %d %d", &itmp1, &itmp2, &itmp3, &itmp4, &itmp5, &itmp6, &bStart, &bEnd);
		bStart -= elec->bStart_dm; bEnd -= elec->bStart_dm;
		if (ionode) printf("bStart = %d bEnd = %d\n", bStart, bEnd);
	}
	nb = bEnd - bStart;
	fclose(fp);
}
void electronphonon::get_nkpair(){
	if (ionode) printf("\nread ldbd_size.dat to get nkpair_glob:\n");
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char s[200];
	fgets(s, sizeof s, fp);
	fgets(s, sizeof s, fp);
	fgets(s, sizeof s, fp);
	if (fgets(s, sizeof s, fp) != NULL){
		if (!isHole){
			sscanf(s, "%d", &nkpair_glob); if (ionode) printf("nkpair_glob = %d\n", nkpair_glob);
		}
		else{
			int itmp;
			sscanf(s, "%d %d", &itmp, &nkpair_glob); if (ionode) printf("nkpair_glob = %d\n", nkpair_glob);
		}
	}
	fclose(fp);
}
void electronphonon::set_kpair(){
	if (code == "jdftx")
		read_ldbd_kpair();
}
void electronphonon::read_ldbd_kpair(){
	if (ionode) printf("\nread ldbd_kpair_k1st(2nd)(_hole).dat:\n");
	FILE *fpk, *fpkp;
	if (!isHole){
		fpk = fopen("ldbd_data/ldbd_kpair_k1st.bin", "rb");
		fpkp = fopen("ldbd_data/ldbd_kpair_k2nd.bin", "rb");
	}
	else{
		fpk = fopen("ldbd_data/ldbd_kpair_k1st_hole.bin", "rb");
		fpkp = fopen("ldbd_data/ldbd_kpair_k2nd_hole.bin", "rb");
	}
	size_t expected_size = nkpair_glob*sizeof(size_t);
	check_file_size(fpk, expected_size, "ldbd_kpair_k1st(_hole).bin size does not match expected size");
	check_file_size(fpkp, expected_size, "ldbd_kpair_k2nd(_hole).bin size does not match expected size");

	int pos = ikpair0_glob * sizeof(size_t);
	fseek(fpk, pos, SEEK_SET);
	fseek(fpkp, pos, SEEK_SET);

	fread(k1st, sizeof(size_t), nkpair_proc, fpk);
	fread(k2nd, sizeof(size_t), nkpair_proc, fpkp);
	fclose(fpk); fclose(fpkp);
}

void electronphonon::set_ephmat(){
	if (code == "jdftx")
		read_ldbd_eph();
}
void electronphonon::read_ldbd_eph(){
	if (ionode) printf("\nread ldbd_P1(2)_(lindblad/conventional)(_hole).dat:\n");
	string fname1 = "ldbd_data/ldbd_P1", fname2 = "ldbd_data/ldbd_P2", suffix;
	if (alg.scatt == "lindblad")
		suffix = "_lindblad";
	else
		suffix = "_conventional";
	if (isHole) suffix += "_hole";
	fname1 += (suffix + ".bin"); fname2 += (suffix + ".bin");

	size_t Psize = (size_t)std::pow(nb, 4);
	size_t expected_size = nkpair_glob*Psize * 2 * sizeof(double);

	FILE *fp1 = fopen(fname1.c_str(), "rb");
	check_file_size(fp1, expected_size, fname1 + " size does not match expected size");
	fseek_bigfile(fp1, ikpair0_glob, Psize * 2 * sizeof(double));
	fread(P1[0], 2 * sizeof(double), nkpair_proc * Psize, fp1);
	fclose(fp1);

	FILE *fp2 = fopen(fname2.c_str(), "rb");
	check_file_size(fp2, expected_size, fname1 + " size does not match expected size");
	fseek_bigfile(fp2, ikpair0_glob, Psize * 2 * sizeof(double));
	fread(P2[0], 2 * sizeof(double), nkpair_proc * Psize, fp2);
	fclose(fp2);
}

void electronphonon::alloc_ephmat(int ikpair0, int ikpair1){
	ikpair0_glob = ikpair0; ikpair1_glob = ikpair1; nkpair_proc = ikpair1 - ikpair0;
	k1st = new size_t[nkpair_proc];
	k2nd = new size_t[nkpair_proc];
	if (alg.summode){
		P1 = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
		P2 = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
		if (!alg.sparseP) P1_next = new complex[(int)std::pow(nb, 4)];
		if (!alg.sparseP) P2_next = new complex[(int)std::pow(nb, 4)];
		if (alg.expt) P1t = new complex[(int)std::pow(nb, 4)];
		if (alg.expt) P2t = new complex[(int)std::pow(nb, 4)];
	}
	else{
		if (alg.scatt == "lindblad"){
			App = alloc_array(nkpair_proc, nm, nb*nb);
			Amm = alloc_array(nkpair_proc, nm, nb*nb);
			Apm = alloc_array(nkpair_proc, nm, nb*nb);
			Amp = alloc_array(nkpair_proc, nm, nb*nb);
		}
		else
			error_message("alg.scatt must be lindblad if not alg.summode");
	}

	dm = alloc_array(nk_glob, nb*nb);
	dm1 = alloc_array(nk_glob, nb*nb);
	ddmdt_eph = alloc_array(nk_glob, nb*nb);
}

void electronphonon::make_map(){
	ij2i = new int[nb*nb]();
	ij2j = new int[nb*nb]();
	int ij = 0;
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++){
		ij2i[ij] = i;
		ij2j[ij] = j;
		ij++;
	}
}