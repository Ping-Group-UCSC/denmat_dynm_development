#include "ElectronPhonon.h"

void electronphonon::evolve_linear(double t, complex** dm, complex** ddmdt_eph){
	MPI_Barrier(MPI_COMM_WORLD);
	/*
	ostringstream convert; convert << mp->myrank;
	convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
	string fname = dir_debug + "debug_eph_evolve.out." + convert.str();
	FILE *fp;
	bool ldebug = fabs(t - 0) < 1e-6 && DEBUG;
	if (ldebug) fp = fopen(fname.c_str(), "w");
	*/
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
	for (int i = 0; i < nb; i++)
		dm[ik_glob][i*nb + i] -= f_eq[ik_glob][i]; // dm is a temporary matrix, it is fine to change it
	zeros(ddmdt_eph, nk_glob, nb*nb);

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];
		bool isIntravellay = latt->isIntravalley(elec->kvec[ik_glob], elec->kvec[ikp_glob]);
		if (isIntravellay && alg.only_intervalley) continue;
		if (!isIntravellay && alg.only_intravalley) continue;
		//if (ldebug) { fprintf(fp, "\nikpair= %d(%d) ik= %d ikp= %d\n", ikpair_local, nkpair_proc, ik_glob, ikp_glob); fflush(fp); }

		if (alg.summode){
			if (!alg.expt){
				compute_ddmdt(dm[ikp_glob], Lscij[ikpair_local], ddmdt_eph[ik_glob]);
				if (ik_glob < ikp_glob)
					compute_ddmdt(dm[ik_glob], Lscji[ikpair_local], ddmdt_eph[ikp_glob]);
			}
			else{
				compute_Pt(t, e[ik_glob], e[ikp_glob], Lscij[ikpair_local], Lsct, false);
				compute_ddmdt(dm[ikp_glob], Lsct, ddmdt_eph[ik_glob]);
				if (ik_glob < ikp_glob){
					compute_Pt(t, e[ikp_glob], e[ik_glob], Lscji[ikpair_local], Lsct, false);
					compute_ddmdt(dm[ik_glob], Lsct, ddmdt_eph[ikp_glob]);
				}
			}
		}
	}

	mp->allreduce(ddmdt_eph, nk_glob, nb*nb, MPI_SUM);
	//if (ldebug) fclose(fp);
}

void electronphonon::compute_ddmdt(complex *dmkp, complex *lsc, complex *ddmdtk){
	zeros(ddmdt_contrib, nb*nb);
	zgemm_interface(ddmdt_contrib, lsc, dmkp, nb*nb, 1, nb*nb);
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++)
		ddmdtk[i*nb + j] += (prefac_eph*0.5) * (ddmdt_contrib[i*nb + j] + conj(ddmdt_contrib[j*nb + i]));
}

inline void electronphonon::compute_Pt(double t, double *ek, double *ekp, complex *P, complex *Pt, bool minus){
	// P1_n3n2,n4n5 = G^+-_n3n4 * conj(G^+-_n2n5) * nq^+-
	// P1_n3n2,n4n5(t) = P1_n3n2,n4n5 * exp[i*t*(e^k_n3 - e^kp_n4 - e^k_n2 + e^kp_n5)]
	// P2_n3n4,n1n5 = G^-+_n1n3 * conj(G^-+_n5n4) * nq^+-
	// P2_n3n4,n1n5(t) = P2_n3n4,n1n5 * exp[i*t*(e^k_n1 - e^kp_n3 - e^k_n5 + e^kp_n4)]
	for (int i1 = 0; i1 < nb; i1++){
		int n1 = i1*nb;
		for (int i2 = 0; i2 < nb; i2++){
			int n12 = (n1 + i2)*nb;
			for (int i3 = 0; i3 < nb; i3++){
				int n123 = (n12 + i3)*nb;
				for (int i4 = 0; i4 < nb; i4++){
					if (!minus) Pt[n123 + i4] = P[n123 + i4] * cis((ek[i1] - ekp[i3] - ek[i2] + ekp[i4])*t);
					else Pt[n123 + i4] = P[n123 + i4] * cis((ek[i3] - ekp[i1] - ek[i4] + ekp[i2])*t);
				}
			}
		}
	}
}