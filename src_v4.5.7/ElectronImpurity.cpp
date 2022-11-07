#include "ElectronImpurity.h"

void electronimpurity::read_ldbd_imp_P(int ik, complex *P1, complex *P2){
	if (ik == 0){ // first k pair
		if (!alg.Pin_is_sparse){
			if (ionode) printf("\nread ldbd_P1(2)_(lindblad/conventional)_D%d_(_hole).dat:\n", iD+1);
			string fname1 = "ldbd_data/ldbd_P1_", fname2 = "ldbd_data/ldbd_P2_", suffix;
			suffix = isHole ? alg.scatt + "_D" + int2str(iD+1) + "_hole" : alg.scatt + "_D" + int2str(iD+1);
			fname1 += (suffix + ".bin"); fname2 += (suffix + ".bin");

			fp1 = fopen(fname1.c_str(), "rb"); fp2 = fopen(fname2.c_str(), "rb");
			size_t expected_size = nkpair_glob * (size_t)nbpow4 * 2 * sizeof(double);
			check_file_size(fp1, expected_size, fname1 + " size does not match expected size");
			check_file_size(fp2, expected_size, fname1 + " size does not match expected size");
			fseek_bigfile(fp1, mp->varstart, nbpow4 * 2 * sizeof(double));
			fseek_bigfile(fp2, mp->varstart, nbpow4 * 2 * sizeof(double));
		}
		else{
			if (ionode) printf("Read sP1 and sP2\n");
			if (sP1 != nullptr) delete sP1;
			if (sP2 != nullptr) delete sP2;
			string suffix = isHole ? alg.scatt + "_D" + int2str(iD+1) + "_hole" : alg.scatt + "_D" + int2str(iD+1);
			sP1 = new sparse2D(mp, "ldbd_data/sP1_" + suffix + "_ns.bin", "ldbd_data/sP1_" + suffix + "_s.bin", "ldbd_data/sP1_" + suffix + "_i.bin", "ldbd_data/sP1_" + suffix + "_j.bin", nb*nb, nb*nb);
			sP2 = new sparse2D(mp, "ldbd_data/sP2_" + suffix + "_ns.bin", "ldbd_data/sP2_" + suffix + "_s.bin", "ldbd_data/sP2_" + suffix + "_i.bin", "ldbd_data/sP2_" + suffix + "_j.bin", nb*nb, nb*nb);
			sP1->read_smat(false); // do_test = false
			sP2->read_smat(false);
		}
	}

	if (!alg.Pin_is_sparse){
		fread(P1, 2 * sizeof(double), nbpow4, fp1);
		fread(P2, 2 * sizeof(double), nbpow4, fp2);
	}
	else{
		sP1->smat[ik]->todense(P1, nb*nb, nb*nb);
		sP2->smat[ik]->todense(P2, nb*nb, nb*nb);
	}
	axbyc(P1, nullptr, nbpow4, c0, complex(fraction, 0));
	axbyc(P2, nullptr, nbpow4, c0, complex(fraction, 0));

	if (ik == mp->varend - mp->varstart - 1){ // last k pair
		if (!alg.Pin_is_sparse){
			fclose(fp1); fclose(fp2);
		}
		else{
			delete sP1; delete sP2; sP1 = nullptr; sP2 = nullptr;
		}
	}
}