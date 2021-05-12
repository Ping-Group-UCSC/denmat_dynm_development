#include "ElectronPhonon.h"

void electronphonon::compute_ddmdt_eq(double** f0_expand){
	complex **dm_expand, **dm1_expand, **ddmdt_expand;
	dm_expand = alloc_array(nk_glob, nb_expand*nb_expand); dm1_expand = alloc_array(nk_glob, nb_expand*nb_expand); ddmdt_expand = alloc_array(nk_glob, nb_expand*nb_expand);

	for (int ik = 0; ik < nk_glob; ik++)
	for (int i = 0; i < nb_expand; i++)
	for (int j = 0; j < nb_expand; j++)
	if (i == j){
		dm_expand[ik][i*nb_expand + j] = f0_expand[ik][i];
		dm1_expand[ik][i*nb_expand + j] = 1 - f0_expand[ik][i];
	}

	evolve_driver(0., dm_expand, dm1_expand, ddmdt_expand, true);

	trunc_copy_arraymat(ddmdt_eq, ddmdt_expand, nk_glob, nb_expand, bStart, bEnd);
	dealloc_array(dm_expand); dealloc_array(dm1_expand); dealloc_array(ddmdt_expand);
}

void electronphonon::evolve_driver(double t, complex** dm_expand, complex** dm1_expand, complex** ddmdt_eph_expand, bool compute_eq){
	trunc_copy_arraymat(dm, dm_expand, nk_glob, nb_expand, bStart, bEnd);
	trunc_copy_arraymat(dm1, dm1_expand, nk_glob, nb_expand, bStart, bEnd);
	zeros(ddmdt_eph, nk_glob, nb*nb);

	if (!alg.linearize) evolve(t, dm, dm1, ddmdt_eph);
	else evolve_linear(t, dm, ddmdt_eph);

	zeros(ddmdt_eph_expand, nk_glob, nb_expand*nb_expand);
	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
		set_mat(ddmdt_eph_expand[ik_glob], ddmdt_eph[ik_glob], nb_expand, bStart, bEnd, bStart, bEnd);
}

void electronphonon::evolve(double t, complex** dm, complex** dm1, complex** ddmdt_eph, bool compute_eq){
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
	complex **Pdm = alloc_array(nk_glob, nb*nb), **dm1P = alloc_array(nk_glob, nb*nb), **dPdm, **dm1dP;
	if (!compute_eq && alg.linearize_dPee) { dPdm = alloc_array(nk_glob, nb*nb); dm1dP = alloc_array(nk_glob, nb*nb); }

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];
		bool isIntravellay = latt->isIntravalley(elec->kvec[ik_glob], elec->kvec[ikp_glob]);
		if (isIntravellay && alg.only_intervalley) continue;
		if (!isIntravellay && alg.only_intravalley) continue;

		int iv1 = latt->whichvalley(elec->kvec[ik_glob]);
		int iv2 = latt->whichvalley(elec->kvec[ikp_glob]);
		if (iv1 >=0 && iv2 >=0 && !latt->vtrans[iv1][iv2]) continue;
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
					init_sparse_mat(sP1->smat[ikpair_local], smat1_time, true);
					init_sparse_mat(sP2->smat[ikpair_local], smat2_time, true);
				}
				else{
					compute_sPt(t, e[ik_glob], e[ikp_glob], sP1->smat[ikpair_local], smat1_time, false);
					compute_sPt(t, e[ik_glob], e[ikp_glob], sP2->smat[ikpair_local], smat2_time, true);
				}
				sparse_zgemm(Pdm[ik_glob], true, smat1_time, dm[ikp_glob], nb*nb, 1, nb*nb, c1, c1);
				sparse_zgemm(dm1P[ik_glob], false, smat2_time, dm1[ikp_glob], 1, nb*nb, nb*nb, c1, c1);
				if (ik_glob < ikp_glob){
					init_sparse_mat(smat1_time, sm2_next);
					init_sparse_mat(smat2_time, sm1_next);
					conj(smat1_time->s, sm2_next->s, smat1_time->ns);
					conj(smat2_time->s, sm1_next->s, smat2_time->ns);
					sparse_zgemm(Pdm[ikp_glob], true, sm1_next, dm[ik_glob], nb*nb, 1, nb*nb, c1, c1);
					sparse_zgemm(dm1P[ikp_glob], false, sm2_next, dm1[ik_glob], 1, nb*nb, nb*nb, c1, c1);
				}
			}
			else{
				if (!alg.expt){
					axbyc(P1t, P1[ikpair_local], (int)std::pow(nb, 4));
					axbyc(P2t, P2[ikpair_local], (int)std::pow(nb, 4));
				}
				else{
					compute_Pt(t, e[ik_glob], e[ikp_glob], P1[ikpair_local], P1t, false);
					compute_Pt(t, e[ik_glob], e[ikp_glob], P2[ikpair_local], P2t, true);
				}
				zgemm_interface(Pdm[ik_glob], P1t, dm[ikp_glob], nb*nb, 1, nb*nb, c1, c1);
				zgemm_interface(dm1P[ik_glob], dm1[ikp_glob], P2t, 1, nb*nb, nb*nb, c1, c1);
				if (ik_glob < ikp_glob){
					conj(P1t, P2_next, (int)std::pow(nb, 4));
					conj(P2t, P1_next, (int)std::pow(nb, 4));
					zgemm_interface(Pdm[ikp_glob], P1_next, dm[ik_glob], nb*nb, 1, nb*nb, c1, c1);
					zgemm_interface(dm1P[ikp_glob], dm1[ik_glob], P2_next, 1, nb*nb, nb*nb, c1, c1);
				}
				if (!compute_eq && alg.linearize_dPee){
					for (int i1 = 0; i1 < nb; i1++)
					for (int i2 = 0; i2 < nb; i2++){
						int i12 = i1*nb + i2, n12 = i12*nb*nb;
						for (int i3 = 0; i3 < nb; i3++){
							dPdm[ik_glob][i12] += dP1ee[ikpair_local][n12 + i3*nb + i3] * f_eq[ikp_glob][i3];
							dm1dP[ik_glob][i12] += f1_eq[ikp_glob][i3] * dP2ee[ikpair_local][(i3*nb + i3)*nb*nb + i12];
						}
					}
					if (ik_glob < ikp_glob){
						conj(dP1ee[ikpair_local], P2_next, (int)std::pow(nb, 4));
						conj(dP2ee[ikpair_local], P1_next, (int)std::pow(nb, 4));
						for (int i1 = 0; i1 < nb; i1++)
						for (int i2 = 0; i2 < nb; i2++){
							int i12 = i1*nb + i2, n12 = i12*nb*nb;
							for (int i3 = 0; i3 < nb; i3++){
								dPdm[ikp_glob][i12] += P1_next[n12 + i3*nb + i3] * f_eq[ik_glob][i3];
								dm1dP[ikp_glob][i12] += f1_eq[ik_glob][i3] * P2_next[(i3*nb + i3)*nb*nb + i12];
							}
						}
					}
				}
			}
		}
		else{
			//compute_ddmdt(dm[ik_glob], dm[ikp_glob], dm1[ik_glob], dm1[ikp_glob], App[ikpair_local], Amm[ikpair_local], Apm[ikpair_local], Amp[ikpair_local], ddmdt_eph[ik_glob]);
			error_message("!alg.summode not yet implemented");
		}
	}

	mp->allreduce(Pdm, nk_glob, nb*nb, MPI_SUM); mp->allreduce(dm1P, nk_glob, nb*nb, MPI_SUM);
	if (!compute_eq && alg.linearize_dPee){ mp->allreduce(dPdm, nk_glob, nb*nb, MPI_SUM); mp->allreduce(dm1dP, nk_glob, nb*nb, MPI_SUM); }

	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++){
		zeros(ddmdt_contrib, nb*nb);
		zhemm_interface(ddmdt_contrib, true, dm1[ik_glob], Pdm[ik_glob], nb);
		zhemm_interface(ddmdt_contrib, false, dm[ik_glob], dm1P[ik_glob], nb, cm1, c1);
		if (!compute_eq && alg.linearize_dPee){
			for (int i = 0; i < nb; i++)
			for (int j = 0; j < nb; j++)
				ddmdt_contrib[i*nb + j] += (f1_eq[ik_glob][i] * dPdm[ik_glob][i*nb + j] - dm1dP[ik_glob][i*nb + j] * f_eq[ik_glob][j]) * cis((e[ik_glob][i] - e[ik_glob][j])*t);
		}
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			ddmdt_eph[ik_glob][i*nb + j] = (prefac_eph*0.5) * (ddmdt_contrib[i*nb + j] + conj(ddmdt_contrib[j*nb + i]));
	}

	if (!compute_eq && alg.ddmdteq){
		for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
		if (i==j || alg.picture == "schrodinger")
			ddmdt_eph[ik_glob][i*nb + j] -= ddmdt_eq[ik_glob][i*nb + j];
		else
			ddmdt_eph[ik_glob][i*nb + j] -= (ddmdt_eq[ik_glob][i*nb + j] * cis((e[ik_glob][i] - e[ik_glob][j])*t));
	}

	dealloc_array(Pdm); dealloc_array(dm1P); if (!compute_eq && alg.linearize_dPee){ dealloc_array(dPdm); dealloc_array(dm1dP); }
	//if (ldebug) fclose(fp);
}

// suppose phase is zero at t=0.0
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
inline void electronphonon::init_sparse_mat(sparse_mat *sin, sparse_mat *sout, bool copy_elem){
	sout->i = sin->i; sout->j = sin->j; sout->ns = sin->ns; // sout->s has been allocated and will be rewritten, we should not set sout->s = sin->s
	if (copy_elem)
	for (int is = 0; is < sin->ns; is++)
		sout->s[is] = sin->s[is];
}
inline void electronphonon::compute_sPt(double t, double *ek, double *ekp, sparse_mat *sm, sparse_mat *smt, bool minus){
	init_sparse_mat(sm, smt);
	// notice that P has four band indeces
	for (int is = 0; is < sm->ns; is++){
		int ind1 = sm->i[is], ind2 = sm->j[is],
			i = ij2i[ind1], j = ij2j[ind1], k = ij2i[ind2], l = ij2j[ind2];
		if (!minus) smt->s[is] = sm->s[is] * cis((ek[i] - ekp[k] - ek[j] + ekp[l])*t);
		else smt->s[is] = sm->s[is] * cis((ek[k] - ekp[i] - ek[l] + ekp[j])*t);
	}
}
