#include "electron.h"

std::vector<std::vector<vector3<>>> electron::compute_bsq(){
	std::vector<std::vector<vector3<>>> bsq(k.size(), std::vector<vector3<>>(nBandsSel));
	for (size_t ik = 0; ik < k.size(); ik++){
		//diagMatrix Ek = state_elec[ik].E(bStart, bStop), dfde(nBandsSel);
		diagMatrix Ek = state_elec[ik].E(bStart - bRef, bStop - bRef), dfde(nBandsSel);
		for (int id = 0; id < 3; id++){
			//matrix s = state_elec[ik].S[id](bStart, bStop, bStart, bStop);
			matrix s = state_elec[ik].S[id](bStart - bBot_dm, bStop - bBot_dm, bStart - bBot_dm, bStop - bBot_dm);
			matrix sdeg = degProj(s, Ek, degthr);
			diagMatrix ss = diag(sdeg*sdeg);
			// a^2 + b^2 = 1, a^2 - b^2 = sdeg^2 => b^2 = (1 - sdeg^2) / 2
			for (int b = 0; b < nBandsSel; b++)
				bsq[ik][b][id] = 0.5 * (1 - sqrt(ss[b]));
		}
	}
	if (mpiWorld->isHead()){
		string fname_bsq = "ldbd_data/ldbd_bsq.out";
		FILE *fp_bsq = fopen(fname_bsq.c_str(), "w");
		fprintf(fp_bsq, "E(Har) b^2\n");
		for (size_t ik = 0; ik < k.size(); ik++){
			//diagMatrix Ek = state_elec[ik].E(bStart, bStop);
			diagMatrix Ek = state_elec[ik].E(bStart - bRef, bStop - bRef);
			for (int b = 0; b < nBandsSel; b++)
				fprintf(fp_bsq, "%14.7le %14.7le %14.7le %14.7le\n", Ek[b], bsq[ik][b][0], bsq[ik][b][1], bsq[ik][b][2]);
		}
		fclose(fp_bsq);
	}
	//MPI_Barrier(MPI_COMM_WORLD);
	return bsq;
}
