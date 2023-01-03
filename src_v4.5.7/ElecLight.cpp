#include "ElecLight.h"

pumpprobeParameters pmp;

void electronlight::evolve_laser(double t, complex** dm, complex** dm1, complex** ddmdt_laser){
	if (pmp.laserAlg == "lindblad")
		evolve_laser_lindblad(t, dm, dm1, ddmdt_laser);
	else if (pmp.laserAlg == "coherent")
		evolve_laser_coh(t, dm, dm1, ddmdt_laser);
}
inline void electronlight::compute_laserPt(double t, complex *Pk, double *ek){
	for (int i = 0; i < nb_dm; i++)
	for (int j = 0; j < nb_dm; j++)
	if (alg.expt_elight)
		laserPt[i*nb_dm + j] = Pk[i*nb_dm + j] * cis((ek[i] - ek[j])*t);
	else
		laserPt[i*nb_dm + j] = Pk[i*nb_dm + j];
}
inline void electronlight::compute_laserPt_coh(double t, complex *Pk, double *ek){
	for (int i = 0; i < nb_dm; i++)
	for (int j = i; j < nb_dm; j++) // only upper triangle part is needed
	if (alg.picture == "interaction" && i != j)
		laserPt[i*nb_dm + j] = Pk[i*nb_dm + j] * cis((ek[i] - ek[j] - pmp.laserE)*t) + Pk[j*nb_dm + i].conj() * cis((ek[i] - ek[j] + pmp.laserE)*t);
	else
		laserPt[i*nb_dm + j] = Pk[i*nb_dm + j] * cis(-pmp.laserE*t) + Pk[j*nb_dm + i].conj() * cis(pmp.laserE*t);
}
void electronlight::evolve_laser_coh(double t, complex** dm, complex** dm1, complex** ddmdt_laser){
	//double trel = t - pmp.pump_tcenter;
	//complex prefac = cmi * pmp.laserA * exp( - std::pow(trel / pmp.pumpTau, 2) / 2) / sqrt(sqrt(M_PI)*pmp.pumpTau);
	complex prefac = cmi * pmp.laserA;
	if (pmp.laserMode == "pump"){
		double trel = t - pmp.pump_tcenter;
		prefac = prefac * exp(-std::pow(trel / pmp.pumpTau, 2) / 2) / sqrt(sqrt(M_PI)*pmp.pumpTau);
	}
	zeros(ddmdt_laser, nk_glob, nb_dm*nb_dm);

	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		compute_laserPt_coh(t, laserP[ik_local], e_dm[ik_glob]);

		// H * dm - dm * H
		//zhemm_interface(ddmdt_laser[ik_glob], false, laserPt, dm[ik_glob], nb_dm);
		//zhemm_interface(ddmdt_laser[ik_glob], true, laserPt, dm[ik_glob], nb_dm, c1, cm1);
		
		//for (int i = 0; i < nb_dm; i++)
		//for (int j = i; j < nb_dm; j++) // only upper triangle part is needed
		//	ddmdt_laser[ik_glob][i] *= prefac;

		// ddm / dt = -i (H * dm - dm * H)
		zhemm_interface(ddmdt_contrib, true, laserPt, dm[ik_glob], nb_dm);
		for (int i = 0; i < nb_dm; i++)
		for (int j = i; j < nb_dm; j++){ // only upper triangle part is needed
			ddmdt_contrib[i*nb_dm + j] -= ddmdt_contrib[j*nb_dm + i].conj(); // (dm * H)^dagger = H * dm
			ddmdt_laser[ik_glob][i*nb_dm + j] = prefac * ddmdt_contrib[i*nb_dm + j];
			ddmdt_laser[ik_glob][j*nb_dm + i] = ddmdt_laser[ik_glob][i*nb_dm + j].conj(); // ddmdt is hermite
		}
	}

	mp->allreduce(ddmdt_laser, nk_glob, nb_dm*nb_dm, MPI_SUM);
}
void electronlight::evolve_laser_lindblad(double t, complex** dm, complex** dm1, complex** ddmdt_laser){
	//double trel = t - pmp.pump_tcenter;
	//double prefac = (M_PI*pmp.laserA*pmp.laserA)
	//	* (exp(-(trel*trel) / (pmp.pumpTau*pmp.pumpTau)) / (sqrt(M_PI)*pmp.pumpTau));
	double prefac = M_PI*pmp.laserA*pmp.laserA;
	if (pmp.laserMode == "pump"){
		double trel = t - pmp.pump_tcenter;
		prefac = prefac
			* (exp(-(trel*trel) / (pmp.pumpTau*pmp.pumpTau)) / (sqrt(M_PI)*pmp.pumpTau));
	}
	zeros(ddmdt_laser, nk_glob, nb_dm*nb_dm);

	//(1-rho) P rho + rho P^* (1-rho)
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		zeros(ddmdt_contrib, nb_dm*nb_dm);

		compute_laserPt(t, laserP[ik_local], e_dm[ik_glob]);
		hermite(laserPt, laserPdag, nb_dm);

		term_plus(dm1[ik_glob], laserPt, dm[ik_glob], laserPdag);
		term_minus(laserPdag, dm1[ik_glob], laserPt, dm[ik_glob]);

		term_plus(dm1[ik_glob], laserPdag, dm[ik_glob], laserPt);
		term_minus(laserPt, dm1[ik_glob], laserPdag, dm[ik_glob]);

		for (int i = 0; i < nb_dm; i++)
		for (int j = 0; j < nb_dm; j++)
			ddmdt_laser[ik_glob][i*nb_dm + j] += prefac * (ddmdt_contrib[i*nb_dm + j] + conj(ddmdt_contrib[j*nb_dm + i]));
	}

	mp->allreduce(ddmdt_laser, nk_glob, nb_dm*nb_dm, MPI_SUM);
}
inline void electronlight::term_plus(complex *dm1, complex *a, complex *dm, complex *b){
	// + (1-dm) * a * dm * b
	zhemm_interface(maux1_dm, true, dm, b, nb_dm);
	zgemm_interface(maux2_dm, a, maux1_dm, nb_dm);
	zhemm_interface(maux1_dm, true, dm1, maux2_dm, nb_dm);
	for (int i = 0; i < nb_dm*nb_dm; i++)
		ddmdt_contrib[i] += maux1_dm[i];
}
inline void electronlight::term_minus(complex *a, complex *dm1, complex *b, complex *dm){
	// - a * (1-dm) * b * dm
	zhemm_interface(maux1_dm, false, dm, b, nb_dm);
	zhemm_interface(maux2_dm, true, dm1, maux1_dm, nb_dm);
	zgemm_interface(maux1_dm, a, maux2_dm, nb_dm);
	// notice that this terms is substract
	for (int i = 0; i < nb_dm*nb_dm; i++)
		ddmdt_contrib[i] -= maux1_dm[i];
}

void electronlight::compute_laserP(){
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		// v are not distributed through cores but laserP are
		for (int iDir = 0; iDir < 3; iDir++)
			trunc_copy_mat(v_dm[iDir], v[ik_glob][iDir], nb, 0, nb_dm, bStart_dm, bEnd_dm);
		vec3_dot_vec3array(laserP[ik_local], pmp.laserPol, v_dm, nb_dm*nb_dm);
	}

	if (pmp.laserAlg == "perturb" || pmp.laserAlg == "lindblad"){ // not needed for pmp.laserAlg == "coherent"
		double prefac_exp = -0.5*pmp.pumpTau*pmp.pumpTau,
			prefac_delta = sqrt(pmp.pumpTau / sqrt(M_PI));

		for (int ik_local = 0; ik_local < nk_proc; ik_local++){
			int ik_glob = ik_local + ik0_glob;
			for (int i = 0; i < nb_dm; i++)
			for (int j = 0; j < nb_dm; j++){
				double de = e_dm[ik_glob][i] - e_dm[ik_glob][j] - pmp.laserE;
				laserP[ik_local][i*nb_dm + j] *= (prefac_delta * exp(prefac_exp * de*de));
			}
		}
	}
}

void electronlight::pump_pert(){
	if (ionode) printf("\nenter pump_pert\n");
	zeros(dm_pump, nk_glob, nb_dm*nb_dm);
	double prefac = M_PI * pmp.laserA * pmp.laserA;

	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;
		zeros(deltaRho, nb_dm*nb_dm);

		hermite(laserP[ik_local], laserPdag, nb_dm); // void hermite(complex *m, complex *h, int n); in mymatrix.h

		term_plus(fbar_dm[ik_glob], laserP[ik_local], f_dm[ik_glob], laserPdag);
		term_minus(laserPdag, fbar_dm[ik_glob], laserP[ik_local], f_dm[ik_glob]);

		term_plus(fbar_dm[ik_glob], laserPdag, f_dm[ik_glob], laserP[ik_local]);
		term_minus(laserP[ik_local], fbar_dm[ik_glob], laserPdag, f_dm[ik_glob]);

		for (int i = 0; i < nb_dm; i++)
		for (int j = 0; j < nb_dm; j++)
			dm_pump[ik_glob][i*nb_dm + j] += prefac * (deltaRho[i*nb_dm + j] + conj(deltaRho[j*nb_dm + i]));
	}

	mp->allreduce(dm_pump, nk_glob, nb_dm*nb_dm, MPI_SUM);

	for (int ik_glob = 0; ik_glob < nk_glob; ik_glob++)
	for (int i = 0; i < nb_dm; i++)
		dm_pump[ik_glob][i*nb_dm + i] += f_dm[ik_glob][i];
}
inline void electronlight::term_plus(double *d1, complex *m1, double *d2, complex *m2){
	mat_diag_mult(maux1_dm, d2, m2, nb_dm);
	zgemm_interface(maux2_dm, m1, maux1_dm, nb_dm);
	//zhemm_interface(maux2, true, m1, maux1_dm, nb_dm);
	mat_diag_mult(maux1_dm, d1, maux2_dm, nb_dm);
	for (int i = 0; i < nb_dm*nb_dm; i++)
		deltaRho[i] += maux1_dm[i];
}
inline void electronlight::term_minus(complex *m1, double *d1, complex *m2, double *d2){
	mat_diag_mult(maux1_dm, m2, d2, nb_dm);
	mat_diag_mult(maux2_dm, d1, maux1_dm, nb_dm);
	zgemm_interface(maux1_dm, m1, maux2_dm, nb_dm);
	//zhemm_interface(maux1_dm, true, m1, maux2_dm, nb_dm);
	for (int i = 0; i < nb_dm*nb_dm; i++)
		deltaRho[i] -= maux1_dm[i];
}

void electronlight::calcImEps(double t, complex **dm, complex **dm1){
	double prefac_const = 4 * std::pow(M_PI, 2) / nk_full;
	if (latt->dim == 2) prefac_const /= (latt->area * latt->thickness);
	else prefac_const /= latt->cell_size;
	double prefac_exp = -0.5*pmp.probeTau*pmp.probeTau,
		prefac_delta = sqrt(pmp.probeTau / sqrt(M_PI));
	zeros(imEps, pmp.probePol.size(), pmp.probeNE);
	
	// compute imEps
	for (int ik_local = 0; ik_local < nk_proc; ik_local++){
		int ik_glob = ik_local + ik0_glob;

		expand_denmat(ik_glob, dm[ik_glob], dm1[ik_glob]);

		// expand v = v_full[bStart_dm:bEnd_dm, 0:nb] to v_full using the hermmiticity of v_full
		for (int iPol = 0; iPol < pmp.probePol.size(); iPol++){
			zeros(v_full, 3, nb*nb);
			for (int iDir = 0; iDir < 3; iDir++){
				set_mat(v_full[iDir], v[ik_glob][iDir], nb, bStart_dm, bEnd_dm, 0, nb);
				hermite(v[ik_glob][iDir], vdag, nb_dm, nb);
				set_mat(v_full[iDir], vdag, nb, 0, nb, bStart_dm, bEnd_dm);
			}
			vec3_dot_vec3array(probePpol[iPol], pmp.probePol[iPol], v_full, nb*nb);
		}

		for (int ie = 0; ie < pmp.probeNE; ie++){
			double probeE = pmp.probeEmin + ie * pmp.probeDE;
			double prefac = prefac_const / std::pow(std::max(probeE, 1. / pmp.probeTau), 3);

			for (int i = 0; i < nb; i++)
			for (int j = 0; j < nb; j++){
				double de = e[ik_glob][i] - e[ik_glob][j] - probeE;
				delta[i*nb + j] = prefac_delta * exp(prefac_exp * de*de);
			}

			for (int iPol = 0; iPol < pmp.probePol.size(); iPol++){
				zeros(deltaf, nb);

				for (int i = 0; i < nb; i++)
				for (int j = 0; j < nb; j++)
					probeP[i*nb + j] = probePpol[iPol][i*nb + j] * delta[i*nb + j];
				compute_probePt(t, probeP, e[ik_glob]);
				hermite(probePt, probePdag, nb); // void hermite(complex *m, complex *h, int n); in mymatrix.h

				term_plus_probe(dm1_expand, probePt, dm_expand, probePdag);
				term_minus_probe(probePdag, dm1_expand, probePt, dm_expand);

				term_plus_probe(dm1_expand, probePdag, dm_expand, probePt);
				term_minus_probe(probePt, dm1_expand, probePdag, dm_expand);
				
				imEps[iPol][ie] += prefac * dot(e[ik_glob], deltaf, nb);
			}
		}
	}

	mp->allreduce(imEps, pmp.probePol.size(), pmp.probeNE, MPI_SUM);
}
inline void electronlight::expand_denmat(int ik_glob, complex *dm, complex *dm1){
	for (int i = 0; i < nb; i++){
		dm_expand[i*nb + i] = f[ik_glob][i];
		dm1_expand[i*nb + i] = 1 - f[ik_glob][i];
	}
	set_mat(dm_expand, dm, nb, bStart_dm, bEnd_dm, bStart_dm, bEnd_dm);
	set_mat(dm1_expand, dm1, nb, bStart_dm, bEnd_dm, bStart_dm, bEnd_dm);
}
inline void electronlight::compute_probePt(double t, complex *Pk, double *ek){
	for (int i = 0; i < nb; i++)
	for (int j = 0; j < nb; j++)
	if (alg.expt_elight)
		probePt[i*nb + j] = Pk[i*nb + j] * cis((ek[i] - ek[j])*t);
	else
		probePt[i*nb + j] = Pk[i*nb + j];
}
inline void electronlight::term_plus_probe(complex *dm1, complex *a, complex *dm, complex *b){
	// + (1-dm) * a * dm * b
	zhemm_interface(maux1, true, dm, b, nb);
	zgemm_interface(maux2, a, maux1, nb);
	zhemm_interface(maux1, true, dm1, maux2, nb);
	for (int i = 0; i < nb; i++)
		deltaf[i] += real(maux1[i*nb + i]);
}
inline void electronlight::term_minus_probe(complex *a, complex *dm1, complex *b, complex *dm){
	// - a * (1-dm) * b * dm
	zhemm_interface(maux1, false, dm, b, nb);
	zhemm_interface(maux2, true, dm1, maux1, nb);
	zgemm_interface(maux1, a, maux2, nb);
	// notice that this terms is substract
	for (int i = 0; i < nb; i++)
		deltaf[i] -= real(maux1[i*nb + i]);
}

void electronlight::write_imEpsVSomega(string fname){
	FILE *fp = fopen(fname.c_str(), "w");
	fprintf(fp, "# omega(eV)");
	for (int iPol = 0; iPol < pmp.probePol.size(); iPol++)
		fprintf(fp, "  %s", pmp.probePoltype[iPol].c_str());
	fprintf(fp, "\n");
	for (int ie = 0; ie < pmp.probeNE; ie++){
		double probeE = pmp.probeEmin + ie * pmp.probeDE;
		fprintf(fp, "%11.7lf", probeE / eV);
		for (int iPol = 0; iPol < pmp.probePol.size(); iPol++)
			fprintf(fp, " %21.14le", imEps[iPol][ie]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void electronlight::probe(int it, double t, complex **dm, complex **dm1){
	if (pmp.probePol.size() == 0 || pmp.probeNE == 0) return;

	calcImEps(t, dm, dm1);

	if (it < 0){
		if (ionode) printf("\nprobe ground state\n");
		trunc_copy_array(imEpsGS, imEps, pmp.probePol.size(), 0, pmp.probeNE); // just copy imEps to imEpsGS
	}
	else{
		for (int iPol = 0; iPol < pmp.probePol.size(); iPol++)
		for (int ie = 0; ie < pmp.probeNE; ie++)
			imEps[iPol][ie] -= imEpsGS[iPol][ie];
	}

	// open imEps files
	if (ionode){
		string fname_it;
		if (it < 0) fname_it = "probe_results/imEpsGS.dat";
		else fname_it = "probe_results/imEps." + int2str(it + it_start) + ".dat";
		//else{
		//	ostringstream ossIt; ossIt << it + it_start;
		//	fname_it = "probe_results/imEps." + ossIt.str() + ".dat";
		//}
		write_imEpsVSomega(fname_it);
	}
}