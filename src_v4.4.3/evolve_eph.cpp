#include "ElectronPhonon.h"

void electronphonon::compute_ddm_eq(double** f0_expand){
	complex **dm_expand, **dm1_expand, **ddm_expand;
	dm_expand = alloc_array(nk_glob, nb_expand*nb_expand); dm1_expand = alloc_array(nk_glob, nb_expand*nb_expand); ddm_expand = alloc_array(nk_glob, nb_expand*nb_expand);

	for (int ik = 0; ik < nk_glob; ik++)
	for (int i = 0; i < nb_expand; i++)
	for (int j = 0; j < nb_expand; j++)
	if (i == j){
		dm_expand[ik][i*nb_expand + j] = f0_expand[ik][i];
		dm1_expand[ik][i*nb_expand + j] = 1 - f0_expand[ik][i];
	}

	evolve_driver(0., dm_expand, dm1_expand, ddm_expand);

	trunc_copy_arraymat(ddm_eq, ddm_expand, nk_glob, nb_expand, bStart, bEnd);
	dealloc_array(dm_expand); dealloc_array(dm1_expand); dealloc_array(ddm_expand);
}

void electronphonon::evolve_driver(double t, complex** dm_expand, complex** dm1_expand, complex** ddmdt_eph_expand){
	trunc_copy_arraymat(dm, dm_expand, nk_glob, nb_expand, bStart, bEnd);
	trunc_copy_arraymat(dm1, dm1_expand, nk_glob, nb_expand, bStart, bEnd);
	zeros(ddmdt_eph, nk_glob, nb*nb);

	evolve(t, dm, dm1, ddmdt_eph);

	zeros(ddmdt_eph_expand, nk_glob, nb_expand*nb_expand);
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
		set_mat(ddmdt_eph_expand[ik_glob], ddmdt_eph[ik_glob], nb_expand, bStart, bEnd, bStart, bEnd);
	//for (int i = 0; i < nb; i++){
	//	int i_expand = i + bStart;
	//	for (int j = 0; j < nb; j++){
	//		int j_expand = j + bStart;
	//		ddmdt_eph_expand[ik_glob][i_expand*nb_expand + j_expand] = ddmdt_eph[ik_glob][i*nb + j];
	//	}
	//}
}

void electronphonon::evolve(double t, complex** dm, complex** dm1, complex** ddmdt_eph){
	MPI_Barrier(MPI_COMM_WORLD);
	/*
	ostringstream convert; convert << mp->myrank;
	convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
	string fname = dir_debug + "debug_eph_evolve.out." + convert.str();
	FILE *fp;
	bool ldebug = fabs(t - 0) < 1e-6 && DEBUG;
	if (ldebug) fp = fopen(fname.c_str(), "w");
	*/
	// dm1 = 1 - dm;
	zeros(ddmdt_eph, nk_glob, nb*nb);

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];
		bool isIntravellay = latt->isIntravalley(elec->kvec[ik_glob], elec->kvec[ikp_glob]);
		if (isIntravellay && alg.only_intervalley) continue;
		if (!isIntravellay && alg.only_intravalley) continue;
		//if (ldebug) { fprintf(fp, "\nikpair= %d(%d) ik= %d ikp= %d\n", ikpair_local, nkpair_proc, ik_glob, ikp_glob); fflush(fp); }

		if (alg.summode){
			// ddmdt = pi/Nq Re { (1-dm)^k_n1n3 P1^kk'_n3n2,n4n5 dm^k'_n4n5
			//                  - (1-dm)^k'_n3n4 P2^kk'_n3n4,n1n5 dm^k_n5n2 + H.C.
			// P1_n3n2,n4n5 = G^+-_n3n4 * conj(G^+-_n2n5) * nq^+-
			// P2_n3n4,n1n5 = G^-+_n1n3 * conj(G^-+_n5n4) * nq^+-

			//if (ldebug) { fprintf_complex_mat(fp, P1[ikpair_local], nb*nb, "P1[ikpair_local]:"); fflush(fp); }
			//if (ldebug) { fprintf_complex_mat(fp, P2[ikpair_local], nb*nb, "P2[ikpair_local]:"); fflush(fp); }

			if (alg.Pin_is_sparse || alg.sparseP){
				if (!alg.expt){
					compute_ddm(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], sP1->smat[ikpair_local], sP2->smat[ikpair_local], ddmdt_eph[ik_glob]);
					if (ik_glob < ikp_glob){
						init_sparse_mat(sP1->smat[ikpair_local], sm2_next);
						init_sparse_mat(sP2->smat[ikpair_local], sm1_next);
						conj(sP1->smat[ikpair_local]->s, sm2_next->s, sP1->smat[ikpair_local]->ns);
						conj(sP2->smat[ikpair_local]->s, sm1_next->s, sP2->smat[ikpair_local]->ns);
						compute_ddm(dm[ikp_glob], dm[ik_glob], dm1[ikp_glob], dm1[ik_glob], sm1_next, sm2_next, ddmdt_eph[ikp_glob]); // swap k and kp
					}
				}
				else{
					compute_sPt(t, sP1->smat[ikpair_local], sP2->smat[ikpair_local], e[ik_glob], e[ikp_glob]);
					compute_ddm(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], smat1_time, smat2_time, ddmdt_eph[ik_glob]);
					if (ik_glob < ikp_glob){
						init_sparse_mat(smat1_time, sm2_next);
						init_sparse_mat(smat2_time, sm1_next);
						conj(smat1_time->s, sm2_next->s, smat1_time->ns);
						conj(smat2_time->s, sm1_next->s, smat2_time->ns);
						compute_ddm(dm[ikp_glob], dm[ik_glob], dm1[ikp_glob], dm1[ik_glob], sm1_next, sm2_next, ddmdt_eph[ikp_glob]); // swap k and kp
					}
				}
			}
			else{
				if (!alg.expt){
					compute_ddm(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], P1[ikpair_local], P2[ikpair_local], ddmdt_eph[ik_glob]);
					if (ik_glob < ikp_glob){
						conj(P1[ikpair_local], P2_next, (int)std::pow(nb, 4));
						conj(P2[ikpair_local], P1_next, (int)std::pow(nb, 4));
						compute_ddm(dm[ikp_glob], dm[ik_glob], dm1[ikp_glob], dm1[ik_glob], P1_next, P2_next, ddmdt_eph[ikp_glob]); // swap k and kp
					}
				}
				else{
					compute_Pt(t, P1[ikpair_local], P2[ikpair_local], e[ik_glob], e[ikp_glob]);
					compute_ddm(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], P1t, P2t, ddmdt_eph[ik_glob]);
					if (ik_glob < ikp_glob){
						conj(P1t, P2_next, (int)std::pow(nb, 4));
						conj(P2t, P1_next, (int)std::pow(nb, 4));
						compute_ddm(dm[ikp_glob], dm[ik_glob], dm1[ikp_glob], dm1[ik_glob], P1_next, P2_next, ddmdt_eph[ikp_glob]); // swap k and kp
					}
				}
			}
		}
		else
			compute_ddm(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], App[ikpair_local], Amm[ikpair_local], Apm[ikpair_local], Amp[ikpair_local], ddmdt_eph[ik_glob]);
	}

	mp->allreduce(ddmdt_eph, nk_glob, nb*nb, MPI_SUM);

	if (alg.ddmeq){
		for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
		if (i==j || alg.picture == "schrodinger")
			ddmdt_eph[ik_glob][i*nb + j] -= ddm_eq[ik_glob][i*nb + j];
		else
			ddmdt_eph[ik_glob][i*nb + j] -= (ddm_eq[ik_glob][i*nb + j] * cis((e[ik_glob][i] - e[ik_glob][j])*t));
	}

	//if (ldebug) fclose(fp);
}
void electronphonon::compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, complex *p1, complex *p2, complex *ddmk){
	zeros(ddmdt_contrib, nb*nb);
	term1_P(dm1k, p1, dmkp);
	term2_P(dm1kp, p2, dmk);
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++)
		ddmk[i*nb + j] += (prefac_eph*0.5) * (ddmdt_contrib[i*nb + j] + conj(ddmdt_contrib[j*nb + i]));
}
void electronphonon::compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, sparse_mat *sm1, sparse_mat *sm2, complex *ddmk){
	zeros(ddmdt_contrib, nb*nb);
	term1_sP(dm1k, sm1, dmkp);
	term2_sP(dm1kp, sm2, dmk);
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++)
		ddmk[i*nb + j] += (prefac_eph*0.5) * (ddmdt_contrib[i*nb + j] + conj(ddmdt_contrib[j*nb + i]));
}
void electronphonon::compute_ddm(complex *dmk, complex *dmkp, complex *dm1k, complex *dm1kp, complex **App, complex **Amm, complex **Apm, complex **Amp, complex *ddmk){
	zeros(ddmdt_contrib, nb*nb);
	for (int im = 0; im < nm; im++)
		if (alg.scatt == "lindblad"){
			term1(dm1k, App[im], dmkp, App[im]); // + (1-dm_k) * A+_kk' * dm_k' * A+_kk'^dagger
			term1(dm1k, Amm[im], dmkp, Amm[im]); // + (1-dm_k) * A-_kk' * dm_k' * A-_kk'^dagger
			term2(Apm[im], dm1kp, Apm[im], dmk); // - A+_kk' * (1-dm_k') * A+_kk'^dagger * dm_k
			term2(Amp[im], dm1kp, Amp[im], dmk); // - A-_kk' * (1-dm_k') * A-_kk'^dagger * dm_k
		}
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++)
		ddmk[i*nb + j] += (prefac_eph*0.5) * (ddmdt_contrib[i*nb + j] + conj(ddmdt_contrib[j*nb + i]));
}

inline void electronphonon::term1_sP(complex *dm1, sparse_mat *sm, complex *dm){
	// + (1-dm_k)_n1n3 * P1_kk'_n3n2,n4n5 * dm_k'_n4n5
	sparse_zgemm(maux1, true, sm->s, sm->i, sm->j, sm->ns, dm, nb*nb, 1, nb*nb); // maux1 = P1_kk'_n3n2,n4n5 * dm_k'_n4n5
	zhemm_interface(maux2, true, dm1, maux1, nb); // maux2 = (1-dm_k)_n1n3 * maux1
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] += maux2[i];
}
inline void electronphonon::term2_sP(complex *dm1, sparse_mat *sm, complex *dm){
	// - (1-dm_k')_n3n4 * P2_kk'_n3n4,n1n5 * dm_k_n5n2
	sparse_zgemm(maux1, false, sm->s, sm->i, sm->j, sm->ns, dm1, 1, nb*nb, nb*nb); // maux1 = (1-dm_k')_n3n4 * P2_kk'_n3n4,n1n5
	zhemm_interface(maux2, false, dm, maux1, nb); // maux2 = maux1 * dm_k_n5n2
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] -= maux2[i];
}
inline void electronphonon::term1_P(complex *dm1, complex *p, complex *dm){
	// + (1-dm_k)_n1n3 * P1_kk'_n3n2,n4n5 * dm_k'_n4n5
	zgemm_interface(maux1, p, dm, nb*nb, 1, nb*nb); // maux1 = P1_kk'_n3n2,n4n5 * dm_k'_n4n5
	zhemm_interface(maux2, true, dm1, maux1, nb); // maux2 = (1-dm_k)_n1n3 * maux1
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] += maux2[i];
}
inline void electronphonon::term2_P(complex *dm1, complex *p, complex *dm){
	// - (1-dm_k')_n3n4 * P2_kk'_n3n4,n1n5 * dm_k_n5n2
	zgemm_interface(maux1, dm1, p, 1, nb*nb, nb*nb); // maux1 = (1-dm_k')_n3n4 * P2_kk'_n3n4,n1n5
	zhemm_interface(maux2, false, dm, maux1, nb); // maux2 = maux1 * dm_k_n5n2
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] -= maux2[i];
}
inline void electronphonon::term1(complex *dm1, complex *a, complex *dm, complex *b){
	// + (1-dm_k) * a+-_kk' * dm_k' * b+-_kk'^dagger
	hermite(b, maux2);
	zhemm_interface(maux1, true, dm, maux2, nb); // maux1 = dm_k' * b_kk'^dagger
	zgemm_interface(maux2, a, maux1, nb); // maux2 = a_kk' * maux1
	zhemm_interface(maux1, true, dm1, maux2, nb); // maux1 = (1-dm_k) * maux2
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] += maux1[i];
}
inline void electronphonon::term2(complex *a, complex *dm1, complex *b, complex *dm){
	// - a+_kk' * (1-dm_k') * b+_kk'^dagger * dm_k
	hermite(b, maux2);
	zhemm_interface(maux1, false, dm, maux2, nb); // maux1 = b_kk'^dagger * dm_k
	zhemm_interface(maux2, true, dm1, maux1, nb); // maux2 = (1-dm_k') * maux1
	zgemm_interface(maux1, a, maux2, nb); // maux1 = a_kk' * maux2
	// notice that this terms is substract
	for (int i = 0; i < nb*nb; i++)
		ddmdt_contrib[i] -= maux1[i];
}

// suppose phase is zero at t=0.0
inline void electronphonon::compute_Pt(double t, complex *P1, complex *P2, double *ek, double *ekp){
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
					P1t[n123 + i4] = P1[n123 + i4] * cis((ek[i1] - ekp[i3] - ek[i2] + ekp[i4])*t);
					P2t[n123 + i4] = P2[n123 + i4] * cis((ek[i3] - ekp[i1] - ek[i4] + ekp[i2])*t);
				}
			}
		}
	}
}
inline void electronphonon::init_sparse_mat(sparse_mat *sin, sparse_mat *sout){
	sout->i = sin->i; sout->j = sin->j; sout->ns = sin->ns; // sout->s has been allocated and will be rewritten, we should not set sout->s = sin->s
}
inline void electronphonon::compute_sPt(double t, sparse_mat *sm1, sparse_mat *sm2, double *ek, double *ekp){
	init_sparse_mat(sm1, smat1_time);
	init_sparse_mat(sm2, smat2_time);
	// notice that P has four band indeces
	for (int is1 = 0; is1 < sm1->ns; is1++){
		int ind1 = sm1->i[is1], ind2 = sm1->j[is1],
			i = ij2i[ind1], j = ij2j[ind1], k = ij2i[ind2], l = ij2j[ind2];
		smat1_time->s[is1] = sm1->s[is1] * cis((ek[i] - ekp[k] - ek[j] + ekp[l])*t);
	}
	for (int is2 = 0; is2 < sm2->ns; is2++){
		int ind1 = sm2->i[is2], ind2 = sm2->j[is2],
			i = ij2i[ind1], j = ij2j[ind1], k = ij2i[ind2], l = ij2j[ind2];
		smat2_time->s[is2] = sm2->s[is2] * cis((ek[k] - ekp[i] - ek[l] + ekp[j])*t);
	}
}
