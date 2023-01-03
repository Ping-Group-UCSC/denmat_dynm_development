#include "electron.h"

void electron::deg_proj(complex *m, double *e, int n, double degthr, complex *mdeg){
	for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	if (fabs(e[i] - e[j]) >= degthr) mdeg[i*n + j] = c0;
	else mdeg[i*n + j] = m[i*n + j];
}
double electron::average_dfde(double **arr, double **f, int n1, int n2, bool inv){
	double sum = 0, sum_dfde = 0;
	for (int i1 = 0; i1 < n1; i1++)
	for (int i2 = 0; i2 < n2; i2++){
		double dfde = f[i1][i2] * (1 - f[i1][i2]);
		if (!inv) sum += arr[i1][i2] * dfde;
		else if (fabs(arr[i1][i2]) > 1e-40) sum += 1. / arr[i1][i2] * dfde;
		sum_dfde += dfde;
	}
	return sum / sum_dfde;
}

void electron::compute_b2(double de, double degauss, double degthr){
	compute_b2(de, degauss, degthr, false);
	if (this->rotate_spin_axes) compute_b2(de, degauss, degthr, true);
}

void electron::compute_b2(double de, double degauss, double degthr, bool rotate_spin_axes){
	if (ionode) printf("\n");
	if (ionode) printf("\n**************************************************\n");
	if (ionode) printf("electron: Analysis spin mixing, spin axis, spin-flip angle, etc.:\n");
	if (ionode && rotate_spin_axes){ printf("with rotated spin axes:\n"); sdir_rot.print(stdout, " %lg", true); }
	if (ionode) printf("**************************************************\n");
	if (ionode && !is_dir("electron_analysis")) system("mkdir electron_analysis");

	//rotate the spin axes if needed
	complex ***s_rot = nullptr;
	string str_rot = rotate_spin_axes ? "rotated_" : "";
	if (rotate_spin_axes){
		s_rot = alloc_array(nk, 3, nb_dm*nb_dm);
		for (int ik = mp->varstart; ik < mp->varend; ik++){
			//if (rotate_spin_axes) Sexp = sdir_rot * Sexp;
			for (int id = 2; id >= 0; id--){
				complex *s_rot_kd = s_rot[ik][id];
				for (int jd = 2; jd >= 0; jd--){
					double sdir_rot_ij = sdir_rot(id, jd);
					for (int b1 = 0; b1 < nb_dm; b1++)
					for (int b2 = 0; b2 < nb_dm; b2++)
						s_rot_kd[b1*nb_dm + b2] += sdir_rot_ij * s[ik][jd][b1*nb_dm + b2];
				}
			}
		}
		mp->allreduce(s_rot, nk, 3, nb_dm*nb_dm, MPI_SUM);
	}
	else
		s_rot = s;

	// read electron energy range and maximum phonon energy
	double ebot_dm, etop_dm, omega_max;
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char stmp[200];
	for (int i = 0; i < 8; i++)
		fgets(stmp, sizeof stmp, fp);
	if (fgets(stmp, sizeof stmp, fp) != NULL){
		double dtmp1, dtmp2;
		sscanf(stmp, "%le %le %le %le", &dtmp1, &dtmp2, &ebot_dm, &etop_dm); if (ionode) printf("ebot_dm = %14.7le eV etop_dm = %14.7le eV\n", ebot_dm / eV, etop_dm / eV);
	}
	if (nv_dm > 0 && evmax <= ebot_dm) error_message("evmax <= ebot_dm", "compute_b2");
	if (nb_dm > nv_dm && ecmin >= etop_dm) error_message("ecmin >= etop_dm", "compute_b2");
	if (fgets(stmp, sizeof stmp, fp) != NULL){
		sscanf(stmp, "%le", &omega_max); if (ionode) printf("omega_max = %14.7le meV\n", omega_max / eV * 1000);
	}
	fclose(fp);

	// transition energy grids
	double wstep = 0.001 * eV;
	int nw = round(std::min(omega_max, etop_dm - ebot_dm) / wstep) + 1; // energy step 1 meV
	std::vector<double> wgrid(nw);
	for (int iw = 1; iw < nw; iw++)
		wgrid[iw] = wgrid[iw - 1] + wstep;

	// electron energy grids
	double gap = ecmin - evmax;
	int ne_v, ne_c, ne;
	std::vector<double> egrid;
	if (nv_dm > 0 && nb_dm > nv_dm && gap <= 1.002*de){
		ne = ceil((etop_dm - ebot_dm) / de);
		egrid.resize(ne, 0);
		egrid[0] = ebot_dm;
		for (int ie = 1; ie < ne; ie++)
			egrid[ie] = egrid[ie - 1] + de;
	}
	else{
		ne_v = nv_dm > 0 ? ceil((evmax - ebot_dm) / de) + 1 : 0;
		ne_c = nb_dm > nv_dm ? ceil((etop_dm - ecmin) / de) + 1 : 0;
		ne = ne_v + ne_c;
		egrid.resize(ne, 0);
		if (ne_v > 0) egrid[ne_v - 1] = evmax + 0.501 * de; // shift a little to ensure vbm is closer to egrid[ne_v - 2] than egrid[ne_v - 1]
		for (int ie = ne_v - 2; ie >= 0; ie--)
			egrid[ie] = egrid[ie + 1] - de;
		if (ne_c > 0) egrid[ne_v] = ecmin - 0.501 * de; // shift a little to ensure cbm is closer to egrid[ne_v + 1] than egrid[ne_v]
		for (int ie = ne_v + 1; ie < ne; ie++)
			egrid[ie] = egrid[ie - 1] + de;
	}

	// spin-mixing b2 = 0.5 - s^exp
	double ***b2kn = alloc_real_array(nk, 3, nb_dm);
	complex sdeg[nb_dm*nb_dm]; zeros(sdeg, nb_dm*nb_dm);
	double sisi[nb_dm]; zeros(sisi, nb_dm);
	vector3<double> b2_avg; double sum_dfde = 0;
	for (int ik = 0; ik < nk; ik++){
		for (int id = 2; id >= 0; id--){
			deg_proj(s_rot[ik][id], e_dm[ik], nb_dm, degthr, sdeg);
			aij_bji(sisi, sdeg, sdeg, nb_dm);
			for (int b = 0; b < nb_dm; b++){
				b2kn[ik][id][b] = 0.5 - sqrt(sisi[b]);
				double dfde = f_dm[ik][b] * (1 - f_dm[ik][b]);
				if (id == 2) sum_dfde += dfde;
				b2_avg[id] += b2kn[ik][id][b] * dfde;
			}
		}
	}
	b2_avg = b2_avg / sum_dfde;

	// output spin-mixing
	if (ionode){
		printf("b2 (thermal averaged; method for Kramers degenerate system):\n %lg %lg %lg\n", b2_avg[0], b2_avg[1], b2_avg[2]);
		string sdir[3]; sdir[0] = "x"; sdir[1] = "y"; sdir[2] = "z";
		for (int id = 2; id >= 0; id--){
			string fname = "electron_analysis/b2_" + str_rot + sdir[id] + "_kn.out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "# energy (eV) b2");
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb_dm; b++)
				fprintf(fp, "%14.7le %14.7le\n", e_dm[ik][b] / eV, b2kn[ik][id][b]);
			fclose(fp);
		}
	}

	if (exists("ldbd_data/ldbd_Umat.bin") && nb_eph == nb_dm){
		//*****************************************************************
		// frequency-dependent spin-flip/conserving overlap square and number of spin-flip/conserving transitions are defined as
		// osf2_{kn}(w) = (1/Nsf_{kn}(w)) (1/Nk) sum_{k'n'} |o_{kn,k'n'}|^2 step(spin-flip) delta(e_{kn}-e{k'n'}+-w) 
		// Nsf_{kn}(w) = (1/Nk) sum_{k'n'} step(spin-flip) delta(e_{kn}-e{k'n'}+-w)
		// osc2_{kn}(w) = (1/Nsc_{kn}(w)) (1/Nk) sum_{k'n'} |o_{kn,k'n'}|^2 step(spin-conserving) delta(e_{kn}-e{k'n'}+-w)
		// Nsf_{kn}(w) = (1/Nk) sum_{k'n'} step(spin-flip) delta(e_{kn}-e{k'n'}+-w)
		//*****************************************************************

		double ***osf2ew = alloc_real_array(3, ne, nw), ***Nsfew = alloc_real_array(3, ne, nw),
			***osc2ew = alloc_real_array(3, ne, nw), ***Nscew = alloc_real_array(3, ne, nw);
		double **osf2w_avg = alloc_real_array(3, nw), **Nsfw = alloc_real_array(3, nw),
			**osc2w_avg = alloc_real_array(3, nw), **Nscw = alloc_real_array(3, nw);

		double eig_ik_sdeg[nb_dm], eig_jk_sdeg[nb_dm]; // eigenvalues of energy-degeneracy projections of spin matrices
		complex ovlp[nb_dm*nb_dm], Ui_sdeg[nb_dm*nb_dm], Uj_sdeg[nb_dm*nb_dm], mtmp[nb_dm*nb_dm];
		vector3<double> b2_avg_2;
		std::vector<double> nstates_e(ne);
		size_t count_nk_deg = 0;
		vector3<double> min_ds(1, 1, 1);

		double prefac_gaussexp = -0.5 / std::pow(degauss, 2);
		double max_diff_b2 = 0;
		for (int id = 2; id >= 0; id--){
			for (int ik = mp->varstart; ik < mp->varend; ik++){
				// diagonalize spin matrices in energy-degenerate subspaces
				bool isI_ik = diagonalize_deg(s_rot[ik][id], e_dm[ik], nb_dm, degthr, eig_ik_sdeg, Ui_sdeg);
				if (id == 2 && !isI_ik) count_nk_deg++;
				if (ionode && !isI_ik){
					printf("id= %d ik= %d\n", id, ik);
					printf_real_array(e_dm[ik], nb_dm, "e_dm: ");
					printf_complex_mat(s_rot[ik][id], nb_dm, "s: ");
					printf_real_array(eig_ik_sdeg, nb_dm, "eig_sdeg: ");
					printf_complex_mat(Ui_sdeg, nb_dm, "U_sdeg: ");
				}

				// double check spin-mixing
				for (int b = 0; b < nb_dm; b++){
					double b2_kn_2 = 0.5 - fabs(eig_ik_sdeg[b]);
					if (fabs(b2_kn_2 - b2kn[ik][id][b]) > max_diff_b2) max_diff_b2 = fabs(b2_kn_2 - b2kn[ik][id][b]);
					//if (ionode && fabs(b2_kn_2 - b2kn[ik][id][b]) > 1e-6){
					//	printf("id= %d ik= %d b= %d, b2: %17.10le %17.10le\n", id, ik, b, b2kn[ik][id][b], b2_kn_2);
					//	error_message("|b2_kn_2 - b2kn[ik][id][b]| > 1e-6", "compute_b2");
					//}
					double dfde = f_dm[ik][b] * (1 - f_dm[ik][b]);
					b2_avg_2[id] += b2_kn_2 * dfde;
				}

				// determine energy index of e_dm[ik][b1] in egrid
				std::vector<int> ie1(nb_dm);
				if (id == 2){ // run once
					for (int b1 = 0; b1 < nb_dm; b1++){
						ie1[b1] = b1 < nv_dm ?
							round((e_dm[ik][b1] - egrid[0]) / de) :
							round((e_dm[ik][b1] - egrid[ne_v]) / de) + ne_v;
						if (ie1[b1] >= 0 && ie1[b1] < ne){
							nstates_e[ie1[b1]] += 1;
							if (fabs(e_dm[ik][b1] - egrid[ie1[b1]]) > 0.501*de){
								printf("e_dm[%d][%d]= %14.7le egrid[%d]= %14.7le 0.501*de= %14.7le\n", ik, b1, e_dm[ik][b1], ie1[b1], egrid[ie1[b1]], 0.501*de);
								error_message("|e_dm[ik][b1] - egrid[ie1[b1]]| > 0.501*de", "compute_b2");
							}
						}
					}
				}

				for (int jk = 0; jk < nk; jk++){
					int iv1, iv2;
					if (!latt->kpair_is_allowed(kvec[ik], kvec[jk], iv1, iv2)) continue;
					// diagonalize spin matrices in energy-degenerate subspaces
					bool isI_jk = diagonalize_deg(s_rot[jk][id], e_dm[jk], nb_dm, degthr, eig_jk_sdeg, Uj_sdeg);

					// compute overlap matrix
					zgemm_interface(ovlp, U[ik], U[jk], nb_dm, nb_dm, nb_wannier, c1, c0, CblasConjTrans);
					if (!isI_jk) zgemm_interface(mtmp, ovlp, Uj_sdeg, nb_dm);
					if (!isI_ik) zgemm_interface(ovlp, Ui_sdeg, mtmp, nb_dm, c1, c0, CblasConjTrans);

					for (int b1 = 0; b1 < nb_dm; b1++){
						if (ie1[b1] < 0 || ie1[b1] >= ne) continue;

						for (int b2 = 0; b2 < nb_dm; b2++){
							bool step_sf = eig_ik_sdeg[b1] * eig_jk_sdeg[b2] < 0; // true for a spin-flip transition
							if (step_sf && abs(eig_ik_sdeg[b1] - eig_jk_sdeg[b2]) < min_ds[id])
								min_ds[id] = abs(eig_ik_sdeg[b1] - eig_jk_sdeg[b2]);

							double o2_pair = ovlp[b1*nb_dm + b2].norm(); // scattering spin-mixing

							double f2f1bar = f_dm[jk][b2] * (1 - f_dm[ik][b1]);
							double f1f2bar = f_dm[ik][b1] * (1 - f_dm[jk][b2]);

							double de = e_dm[ik][b1] - e_dm[jk][b2];
							for (int iw = 0; iw < nw; iw++){
								double delta_minus = exp(prefac_gaussexp * std::pow(de - wgrid[iw], 2)); // gaussian delta without prefactor
								double delta_plus = exp(prefac_gaussexp * std::pow(de + wgrid[iw], 2)); // gaussian delta without prefactor
								double weight_o2ew = delta_minus + delta_plus;
								double weight_o2w = f2f1bar * delta_minus + f1f2bar * delta_plus;

								if (step_sf){
									// frequency-dependent spin-flip overlap square and number of spin-flip transitions
									osf2ew[id][ie1[b1]][iw] += o2_pair * weight_o2ew;
									Nsfew[id][ie1[b1]][iw] += weight_o2ew;
									osf2w_avg[id][iw] += o2_pair * weight_o2w;
									Nsfw[id][iw] += weight_o2w;
								}
								else{
									// frequency-dependent spin-conserving overlap square and number of spin-conserving transitions
									osc2ew[id][ie1[b1]][iw] += o2_pair * weight_o2ew;
									Nscew[id][ie1[b1]][iw] += weight_o2ew;
									osc2w_avg[id][iw] += o2_pair * weight_o2w;
									Nscw[id][iw] += weight_o2w;
								}
							}
						}
					}
				}
			}
			// collecting data
			mp->allreduce(osf2ew[id], ne, nw, MPI_SUM); mp->allreduce(Nsfew[id], ne, nw, MPI_SUM);
			mp->allreduce(osc2ew[id], ne, nw, MPI_SUM); mp->allreduce(Nscew[id], ne, nw, MPI_SUM);
			mp->allreduce(b2_avg_2[id], MPI_SUM);
			mp->allreduce(min_ds[id], MPI_MIN);
		}
		// collecting data
		mp->allreduce(osf2w_avg, 3, nw, MPI_SUM); mp->allreduce(Nsfw, 3, nw, MPI_SUM);
		mp->allreduce(osc2w_avg, 3, nw, MPI_SUM); mp->allreduce(Nscw, 3, nw, MPI_SUM);
		mp->allreduce(&nstates_e[0], ne, MPI_SUM);
		mp->allreduce(count_nk_deg, MPI_SUM);
		mp->allreduce(max_diff_b2, MPI_MAX);
		// useful information
		if (ionode){
			printf("no. of k with energy degeneracy (tot no. of k): %lu (%d)\n", count_nk_deg, nk);
			printf("min. spin change along x: %lg\n", min_ds[0]);
			printf("min. spin change along y: %lg\n", min_ds[1]);
			printf("min. spin change along z: %lg\n", min_ds[2]); fflush(stdout);
		}
		// double check spin-mixing again
		b2_avg_2 = b2_avg_2 / sum_dfde;
		for (int id = 2; id >= 0; id--){
			if (ionode) printf("id= %d b2 by two methods (S^2 / diagonalization): %14.7le %14.7le\n", id, b2_avg[id], b2_avg_2[id]);
			//if (ionode && fabs(b2_avg_2[id] - b2_avg[id]) > 1e-6){
			//	printf("id= %d b2: %14.7le %14.7le\n", id, b2_avg[id], b2_avg_2[id]);
			//	error_message("|b2_avg_2[id] - b2_avg[id]| > 1e-6", "compute_b2");
			//}
		}
		// divide normalization factor
		double prefac_gauss = 1. / (sqrt(2 * M_PI) * degauss);
		double prefac_Nw = prefac_gauss / nk_full / nk_full;
		for (int id = 2; id >= 0; id--){
			for (int ie = 0; ie < ne; ie++){
				double prefac_New = prefac_gauss / nk_full / nstates_e[ie];
				for (int iw = 0; iw < nw; iw++){
					osf2ew[id][ie][iw] = osf2ew[id][ie][iw] / Nsfew[id][ie][iw];
					osc2ew[id][ie][iw] = osc2ew[id][ie][iw] / Nscew[id][ie][iw];
					Nsfew[id][ie][iw] *= prefac_New;
					Nscew[id][ie][iw] *= prefac_New;
				}
			}
			for (int iw = 0; iw < nw; iw++){
				osf2w_avg[id][iw] = osf2w_avg[id][iw] / Nsfw[id][iw];
				osc2w_avg[id][iw] = osc2w_avg[id][iw] / Nscw[id][iw];
				Nsfw[id][iw] *= prefac_Nw;
				Nscw[id][iw] *= prefac_Nw;
			}
		}

		// output frequency-dependent spin-flip/conserving overlap square and number of spin-flip/conserving transitions
		if (ionode){
			string sdir[3]; sdir[0] = "x"; sdir[1] = "y"; sdir[2] = "z";
			for (int id = 2; id >= 0; id--){
				string fnamesf = "electron_analysis/osf2w_" + str_rot + sdir[id] + ".out",
					fnamesc = "electron_analysis/osc2w_" + str_rot + sdir[id] + ".out";
				FILE *fpsf = fopen(fnamesf.c_str(), "w"),
					*fpsc = fopen(fnamesc.c_str(), "w");
				fprintf(fpsf, "# transition energy (meV) osf2w Nsfw osf2*Nsf/Nf\n");
				fprintf(fpsc, "# transition energy (meV) osc2w Nscw osf2*Nsf/Nf\n");
				for (int iw = 0; iw < nw; iw++){
					fprintf(fpsf, "%14.7le %14.7le %14.7le %14.7le\n", wgrid[iw] / eV * 1000, osf2w_avg[id][iw], Nsfw[id][iw], osf2w_avg[id][iw] * Nsfw[id][iw] / (sum_dfde / nk_full));
					fprintf(fpsc, "%14.7le %14.7le %14.7le %14.7le\n", wgrid[iw] / eV * 1000, osc2w_avg[id][iw], Nscw[id][iw], osc2w_avg[id][iw] * Nscw[id][iw] / (sum_dfde / nk_full));
				}
				fclose(fpsf); fclose(fpsc);

				for (int iw = 0; iw < nw; iw++){
					string fnamesf = "electron_analysis/osf2ew_" + str_rot + sdir[id] + "_w" + int2str(iw) + ".out",
						fnamesc = "electron_analysis/osc2ew_" + str_rot + sdir[id] + "_w" + int2str(iw) + ".out";
					FILE *fpsf = fopen(fnamesf.c_str(), "w"),
						*fpsc = fopen(fnamesc.c_str(), "w");
					fprintf(fpsf, "# elec. energy (eV) osf2ew Nsfew osf2e*Nsfe (for transition energy: %lg meV)\n", wgrid[iw] / eV * 1000);
					fprintf(fpsc, "# elec. energy (eV) osc2ew Nscew osc2e*Nsce (for transition energy: %lg meV)\n", wgrid[iw] / eV * 1000);
					for (int ie = 0; ie < ne; ie++){
						fprintf(fpsf, "%14.7le %14.7le %14.7le %14.7le\n", egrid[ie] / eV, osf2ew[id][ie][iw], Nsfew[id][ie][iw], osf2ew[id][ie][iw] * Nsfew[id][ie][iw]);
						fprintf(fpsc, "%14.7le %14.7le %14.7le %14.7le\n", egrid[ie] / eV, osc2ew[id][ie][iw], Nscew[id][ie][iw], osc2ew[id][ie][iw] * Nscew[id][ie][iw]);
					}
					fclose(fpsf); fclose(fpsc);
				}
			}
		}

		//deallocate memory
		dealloc_real_array(osf2ew); dealloc_real_array(Nsfew); dealloc_real_array(osf2w_avg); dealloc_real_array(Nsfw);
		dealloc_real_array(osc2ew); dealloc_real_array(Nscew); dealloc_real_array(osc2w_avg); dealloc_real_array(Nscw);
	}
	dealloc_real_array(b2kn);

	if (rotate_spin_axes) dealloc_array(s_rot);



	/////////////////////////////////////////////////////////////////////////
	// averaged spin axis; angle between spins and spin axis; spin-flip angle
	// currently degeneracy is not considered
	/////////////////////////////////////////////////////////////////////////
	if (rotate_spin_axes) return; // the following analysis are unrelated to whether spin axes are rotated, so do not need to repeat
	
	// spin expectation values
	std::vector<std::vector<vector3<>>> Sexp_dir(nk, std::vector<vector3<>>(nb_dm));
	for (int ik = 0; ik < nk; ik++)
	for (int b = 0; b < nb_dm; b++){
		vector3<> Sexp(s[ik][0][b*nb_dm + b].real(), s[ik][1][b*nb_dm + b].real(), s[ik][2][b*nb_dm + b].real());
		double Sexp_abs = sqrt(Sexp.length_squared());
		Sexp_dir[ik][b] = Sexp / Sexp_abs;
		if (fabs(sqrt(Sexp_dir[ik][b].length_squared()) - 1) > 1e-10){
			printf("ik = %d b = %d sqrt(Sexp_dir.length_squared()) = %14.7le\n", ik, b, sqrt(Sexp_dir[ik][b].length_squared()));
			error_message("fabs(sqrt(Sexp_dir.length_squared()) - 1) > 1e-10", "compute_b2");
		}
	}
	if (ionode){
		string sdir[3]; sdir[0] = "x"; sdir[1] = "y"; sdir[2] = "z";
		for (int id = 2; id >= 0; id--){
			string fname = "electron_analysis/Sexp_dir_" + sdir[id] + "_kn.out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "# energy (eV) Sexp_dir\n");
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb_dm; b++){
				fprintf(fp, "%14.7le %14.7le\n", e_dm[ik][b] / eV, Sexp_dir[ik][b][id]);
			}
			fclose(fp);
		}
	}

	//average spin axis
	vector3<double> Saxis_avg(vector3<double>(0,0,0));
	for (int ik = 0; ik < nk; ik++)
	for (int b = 0; b < nb_dm; b++){
		double cos_theta_from_sdir_z = dot(Sexp_dir[ik][b], sdir_z);
		vector3<double> Saxis = cos_theta_from_sdir_z > 0 ? Sexp_dir[ik][b] : -Sexp_dir[ik][b];
		double dfde = f_dm[ik][b] * (1 - f_dm[ik][b]);
		Saxis_avg = Saxis_avg + Saxis * dfde;
	}
	Saxis_avg = Saxis_avg / sum_dfde;
	if (ionode) printf("Saxis_avg: %lg %lg %lg\n", Saxis_avg[0], Saxis_avg[1], Saxis_avg[2]);

	//angle between k-dependent spins and the averaged spin axis
	double sin2_half_theta_avg = 0;
	for (int ik = 0; ik < nk; ik++)
	for (int b = 0; b < nb_dm; b++){
		double cos_theta = fabs(dot(Sexp_dir[ik][b], Saxis_avg));
		double sin2_half_theta = 1 - cos_theta < 0 ? 0 : 0.5 * (1 - cos_theta);
		double dfde = f_dm[ik][b] * (1 - f_dm[ik][b]);
		sin2_half_theta_avg += sin2_half_theta * dfde;
	}
	sin2_half_theta_avg /= sum_dfde;
	if (ionode) printf("sin2_half_theta_avg = %lg\n", sin2_half_theta_avg);

	//spin-flip angle
	// sin^2(theta^{sf}/2) whether theta^{sf} is the spin-flip angle, i.e., the angle between -S^exp at k1 and S^exp at k2
	double **sin2_half_theta_sf_ew = alloc_real_array(ne, nw), **New = alloc_real_array(ne, nw);;
	std::vector<double> sin2_half_theta_sf_w_avg(nw), Nw(nw);
	std::vector<double> nstates_e(ne,0);

	double prefac_gaussexp = -0.5 / std::pow(degauss, 2);

	for (int ik = mp->varstart; ik < mp->varend; ik++){

		// determine energy index of e_dm[ik][b1] in egrid
		std::vector<int> ie1(nb_dm);
		for (int b1 = 0; b1 < nb_dm; b1++){
			ie1[b1] = b1 < nv_dm ?
				round((e_dm[ik][b1] - egrid[0]) / de) :
				round((e_dm[ik][b1] - egrid[ne_v]) / de) + ne_v;
			if (ie1[b1] >= 0 && ie1[b1] < ne){
				nstates_e[ie1[b1]] += 1;
				if (fabs(e_dm[ik][b1] - egrid[ie1[b1]]) > 0.501*de){
					printf("e_dm[%d][%d]= %14.7le egrid[%d]= %14.7le 0.501*de= %14.7le\n", ik, b1, e_dm[ik][b1], ie1[b1], egrid[ie1[b1]], 0.501*de);
					error_message("|e_dm[ik][b1] - egrid[ie1[b1]]| > 0.501*de", "compute_b2");
				}
			}
		}

		for (int jk = 0; jk < nk; jk++){

			for (int b1 = 0; b1 < nb_dm; b1++){
				if (ie1[b1] < 0 || ie1[b1] >= ne) continue;
				vector3<> Sexp_dir_1(Sexp_dir[ik][b1]);

				for (int b2 = 0; b2 < nb_dm; b2++){
					vector3<> Sexp_dir_2(Sexp_dir[jk][b2]);

					double cos_theta = dot(Sexp_dir_1, Sexp_dir_2);
					if (cos_theta >= 0) continue;

					if (1 + cos_theta < -1e-10){
						printf("ik= %d b1= %d jk= %d b2= %d 1+cos_theta = %14.7le\n", ik, b1, jk, b2, 1 + cos_theta);
						error_message("1 + cos_theta < 0", "compute_b2");
					}
					double sin2_half_theta_sf = (1 + cos_theta < 0) ? 0 : 0.5*(1 + cos_theta);

					double f2f1bar = f_dm[jk][b2] * (1 - f_dm[ik][b1]);
					double f1f2bar = f_dm[ik][b1] * (1 - f_dm[jk][b2]);

					double de = e_dm[ik][b1] - e_dm[jk][b2];
					for (int iw = 0; iw < nw; iw++){
						double delta_minus = exp(prefac_gaussexp * std::pow(de - wgrid[iw], 2)); // gaussian delta without prefactor
						double delta_plus = exp(prefac_gaussexp * std::pow(de + wgrid[iw], 2)); // gaussian delta without prefactor
						double weight_ew = delta_minus + delta_plus;
						double weight_w = f2f1bar * delta_minus + f1f2bar * delta_plus;

						sin2_half_theta_sf_ew[ie1[b1]][iw] += sin2_half_theta_sf * weight_ew;
						New[ie1[b1]][iw] += weight_ew;
						sin2_half_theta_sf_w_avg[iw] += sin2_half_theta_sf * weight_w;
						Nw[iw] += weight_w;
					}
				}
			}
		}
	}
	// collecting data
	mp->allreduce(sin2_half_theta_sf_ew, ne, nw, MPI_SUM); mp->allreduce(New, ne, nw, MPI_SUM);
	mp->allreduce(sin2_half_theta_sf_w_avg.data(), nw, MPI_SUM); mp->allreduce(Nw.data(), nw, MPI_SUM);
	mp->allreduce(&nstates_e[0], ne, MPI_SUM);

	// divide normalization factor
	double prefac_gauss = 1. / (sqrt(2 * M_PI) * degauss);
	double prefac_Nw = prefac_gauss / nk_full / nk_full;
	for (int ie = 0; ie < ne; ie++){
		double prefac_New = prefac_gauss / nk_full / nstates_e[ie];
		for (int iw = 0; iw < nw; iw++){
			sin2_half_theta_sf_ew[ie][iw] = sin2_half_theta_sf_ew[ie][iw] / New[ie][iw];
			New[ie][iw] *= prefac_New;
		}
	}
	for (int iw = 0; iw < nw; iw++){
		sin2_half_theta_sf_w_avg[iw] = sin2_half_theta_sf_w_avg[iw] / Nw[iw];
		Nw[iw] *= prefac_Nw;
	}

	// output frequency-dependent spin-flip angle
	if (ionode){
		string fname = "electron_analysis/sin2_half_theta_sf_w.out";
		FILE *fp = fopen(fname.c_str(), "w");
		fprintf(fp, "# transition energy (meV) sin2_half_theta_sf_w Nw sin2_half_theta_sf_w*Nw/Nf\n");
		for (int iw = 0; iw < nw; iw++)
			fprintf(fp, "%14.7le %14.7le %14.7le %14.7le\n", wgrid[iw] / eV * 1000, sin2_half_theta_sf_w_avg[iw], Nw[iw], sin2_half_theta_sf_w_avg[iw] * Nw[iw] / (sum_dfde / nk_full));
		fclose(fp);

		for (int iw = 0; iw < nw; iw++){
			string fname = "electron_analysis/sin2_half_theta_sf_ew_w" + int2str(iw) + ".out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "# elec. energy (eV) sin2_half_theta_sf_ew New sin2_half_theta_sf_ew*New (for transition energy: %lg meV)\n", wgrid[iw] / eV * 1000);
			for (int ie = 0; ie < ne; ie++)
				fprintf(fp, "%14.7le %14.7le %14.7le %14.7le\n", egrid[ie] / eV, sin2_half_theta_sf_ew[ie][iw], New[ie][iw], sin2_half_theta_sf_ew[ie][iw] * New[ie][iw]);
			fclose(fp);
		}
	}

	//deallocate memory
	dealloc_real_array(sin2_half_theta_sf_ew); dealloc_real_array(New);

	// dos
	std::vector<double> dos(ne);
	for (int ie = 0; ie < ne; ie++)
		dos[ie] = nstates_e[ie] / de / nk_full;
	if (ionode){
		FILE *fp = fopen("electron_analysis/dos.out", "w");
		fprintf(fp, "# elec. energy (eV) dos (a.u.)\n");
		for (int ie = 0; ie < ne; ie++)
			fprintf(fp, "%14.7le %14.7le\n", egrid[ie] / eV, dos[ie]);
		fclose(fp);
	}
	double sum_dossq = 0, sum_dos = 0;
	for (int ie = 0; ie < ne; ie++){
		double fe = electron::fermi(temperature, mu, egrid[ie]);
		double dfde = fe * (1 - fe);
		sum_dossq += dfde * dos[ie] * dos[ie];
		sum_dos += dfde * dos[ie];
	}
	if (ionode) printf("effective scattering dos (inaccurate) = %14.7le\n", sum_dossq / sum_dos);
}

void electron::compute_Bin2(){
	compute_Bin2(false);
	if (this->rotate_spin_axes) compute_Bin2(true);
}

void electron::compute_Bin2(bool rotate_spin_axes){
	//analyse internal magnetic field
	if (ionode && !is_dir("electron_analysis")) system("mkdir electron_analysis");
	if (ionode) printf("\n");
	if (ionode) printf("\n**************************************************\n");
	if (ionode){ printf("electron: Analyse internal magnetic fields\n"); }
	if (ionode && rotate_spin_axes){ printf("with rotated spin axes:\n"); sdir_rot.print(stdout, " %lg", true); }
	if (ionode) printf("**************************************************\n");

	// Bin_i = DeltaE * S^exp_i / 2 / |S^exp|^2
	// Bin_SHalf_i = DeltaE * S^exp_i / |S^exp|, which is Bin_i with |S^exp| normalized to 0.5
	std::vector<vector3<>> Binkn(nk*nb_dm), Binkn_SHalf(nk*nb_dm), Binabskn_SHalf(nk*nb_dm);
	std::vector<double> dfde(nk*nb_dm), dekn(nk*nb_dm);
	for (int ik = 0; ik < nk; ik++)
	for (int b = 0; b < nb_dm; b++){
		int b_dft = b + bStart_dm + bskipped_wannier + bskipped_dft;
		dekn[ik*nb_dm + b] = (b_dft % 2 == 0) ? e_dm[ik][b + 1] - e_dm[ik][b] : e_dm[ik][b] - e_dm[ik][b - 1];
		double de_sign = (b_dft % 2 == 0) ? -dekn[ik*nb_dm + b] : dekn[ik*nb_dm + b]; // include sign for Bin
		vector3<> Sexp(s[ik][0][b*nb_dm + b].real(), s[ik][1][b*nb_dm + b].real(), s[ik][2][b*nb_dm + b].real());
		if (rotate_spin_axes) Sexp = sdir_rot * Sexp;
		double Sexp2 = Sexp.length_squared();
		double Sexp_abs = sqrt(Sexp2);
		double two_Sexp2 = 2 * Sexp2;
		dfde[ik*nb_dm + b] = f_dm[ik][b] * (1 - f_dm[ik][b]);
		for (int id = 0; id < 3; id++){
			Binkn[ik*nb_dm + b][id] = de_sign * Sexp[id] / two_Sexp2;
			Binkn_SHalf[ik*nb_dm + b][id] = de_sign * Sexp[id] / Sexp_abs;
			Binabskn_SHalf[ik*nb_dm + b][id] = fabs(de_sign * Sexp[id] / Sexp_abs);
		}
	}
	double de_mean = mean_of_array(dekn.data(), nk*nb_dm, dfde.data());
	double de_sigma = sigma_of_array(dekn.data(), nk*nb_dm, false, de_mean, dfde.data());
	double de2_avg = de_sigma * de_sigma;
	vector3<> Bin_mean = mean_of_(Binkn, dfde.data()); // actually external magnetic field
	vector3<> Bin_sigma = sigma_of_(Binkn, false, Bin_mean, dfde.data());
	vector3<> Bin2_avg(Bin_sigma[0] * Bin_sigma[0], Bin_sigma[1] * Bin_sigma[1], Bin_sigma[2] * Bin_sigma[2]);
	vector3<> Bin_SHalf_mean = mean_of_(Binkn_SHalf, dfde.data()); // actually external magnetic field
	vector3<> Bin_SHalf_sigma = sigma_of_(Binkn_SHalf, false, Bin_SHalf_mean, dfde.data());
	vector3<> Bin2_SHalf_avg(Bin_SHalf_sigma[0] * Bin_SHalf_sigma[0], Bin_SHalf_sigma[1] * Bin_SHalf_sigma[1], Bin_SHalf_sigma[2] * Bin_SHalf_sigma[2]);
	vector3<> Binabs_SHalf_mean = mean_of_(Binabskn_SHalf, dfde.data()); // actually external magnetic field
	vector3<> Binabs_SHalf_sigma = sigma_of_(Binabskn_SHalf, false, Binabs_SHalf_mean, dfde.data());
	vector3<> Binabs2_SHalf_avg(Binabs_SHalf_sigma[0] * Binabs_SHalf_sigma[0], Binabs_SHalf_sigma[1] * Binabs_SHalf_sigma[1], Binabs_SHalf_sigma[2] * Binabs_SHalf_sigma[2]);

	// output internal magnetic field related information
	if (ionode){
		printf("\nDefine B from DeltaE = 2 B S^exp\n");
		printf("Bext: %lg %lg %lg (a.u.)\n", Bin_mean[0], Bin_mean[1], Bin_mean[2]);
		printf("Bext: %lg %lg %lg (Tesla)\n", Bin_mean[0] / Tesla2au, Bin_mean[1] / Tesla2au, Bin_mean[2] / Tesla2au);
		printf("Bin sigma: %lg %lg %lg (Tesla)\n", Bin_sigma[0] / Tesla2au, Bin_sigma[1] / Tesla2au, Bin_sigma[2] / Tesla2au);
		printf("Bin sigma^2: %lg %lg %lg (a.u.)\n", Bin2_avg[0], Bin2_avg[1], Bin2_avg[2]);
		printf("Bin sigma^2: %lg %lg %lg (ps^-2)\n", Bin2_avg[0] * ps*ps, Bin2_avg[1] * ps*ps, Bin2_avg[2] * ps*ps);

		printf("\nDefine B from DeltaE = 2 B S^exp assuming |S^exp| = 0.5\n");
		printf("Bext_SHalf: %lg %lg %lg (a.u.)\n", Bin_SHalf_mean[0], Bin_SHalf_mean[1], Bin_SHalf_mean[2]);
		printf("Bext_SHalf: %lg %lg %lg (Tesla)\n", Bin_SHalf_mean[0] / Tesla2au, Bin_SHalf_mean[1] / Tesla2au, Bin_SHalf_mean[2] / Tesla2au);
		printf("Bin_SHalf sigma: %lg %lg %lg (Tesla)\n", Bin_SHalf_sigma[0] / Tesla2au, Bin_SHalf_sigma[1] / Tesla2au, Bin_SHalf_sigma[2] / Tesla2au);
		printf("Bin_SHalf sigma^2: %lg %lg %lg (a.u.)\n", Bin2_SHalf_avg[0], Bin2_SHalf_avg[1], Bin2_SHalf_avg[2]);
		printf("Bin_SHalf sigma^2: %lg %lg %lg (ps^-2)\n", Bin2_SHalf_avg[0] * ps*ps, Bin2_SHalf_avg[1] * ps*ps, Bin2_SHalf_avg[2] * ps*ps);
		printf("|Bin_SHalf| sigma: %lg %lg %lg (Tesla)\n", Binabs_SHalf_sigma[0] / Tesla2au, Binabs_SHalf_sigma[1] / Tesla2au, Binabs_SHalf_sigma[2] / Tesla2au);
		printf("|Bin_SHalf| sigma^2: %lg %lg %lg (a.u.)\n", Binabs2_SHalf_avg[0], Binabs2_SHalf_avg[1], Binabs2_SHalf_avg[2]);
		printf("|Bin_SHalf| sigma^2: %lg %lg %lg (ps^-2)\n", Binabs2_SHalf_avg[0] * ps*ps, Binabs2_SHalf_avg[1] * ps*ps, Binabs2_SHalf_avg[2] * ps*ps);

		printf("\nAnalyse energy splitting\n");
		printf("DE mean: %lg (a.u.)\n", de_mean);
		printf("DE mean: %lg (Tesla)\n", de_mean / Tesla2au);
		printf("DE sigma: %lg (Tesla)\n", de_sigma / Tesla2au);
		printf("DE sigma^2: %lg (a.u.)\n", de2_avg);
		printf("DE sigma^2: %lg (ps^-2)\n", de2_avg * ps*ps);

		string rotated = rotate_spin_axes ? "rotated_" : "";
		string sdir[3]; sdir[0] = "x"; sdir[1] = "y"; sdir[2] = "z";
		for (int id = 2; id >= 0; id--){
			string fname = "electron_analysis/Bin_" + rotated + sdir[id] + "_kn.out", 
				fname_SHalf = "electron_analysis/Bin_SHalf_" + rotated + sdir[id] + "_kn.out";
			FILE *fp = fopen(fname.c_str(), "w"), *fp_SHalf = fopen(fname_SHalf.c_str(), "w");
			fprintf(fp, "# energy (eV) Bin\n"); fprintf(fp_SHalf, "# energy (eV) Bin\n");
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb_dm; b++){
				fprintf(fp, "%14.7le %14.7le\n", e_dm[ik][b] / eV, Binkn[ik*nb_dm + b][id]);
				fprintf(fp_SHalf, "%14.7le %14.7le\n", e_dm[ik][b] / eV, Binkn_SHalf[ik*nb_dm + b][id]);
			}
			fclose(fp); fclose(fp_SHalf);
		}

		for (int id = 2; id >= 0; id--){
			string fname = "electron_analysis/Bin2_" + rotated + sdir[id] + "_kn.out", 
				fname_SHalf = "electron_analysis/Bin2_SHalf_" + rotated + sdir[id] + "_kn.out";
			FILE *fp = fopen(fname.c_str(), "w"), *fp_SHalf = fopen(fname_SHalf.c_str(), "w");
			fprintf(fp, "# energy (eV) Bin2\n"); fprintf(fp_SHalf, "# energy (eV) Bin2\n");
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb_dm; b++){
				fprintf(fp, "%14.7le %14.7le\n", e_dm[ik][b] / eV, std::pow(Binkn[ik*nb_dm + b][id] - Bin_mean[id], 2));
				fprintf(fp_SHalf, "%14.7le %14.7le\n", e_dm[ik][b] / eV, std::pow(Binkn_SHalf[ik*nb_dm + b][id] - Bin_SHalf_mean[id], 2));
			}
			fclose(fp); fclose(fp_SHalf);
		}
		
		string fname = "electron_analysis/DeltaE2_fluctuation_kn.out", fname_dfde = "electron_analysis/dfde.out";
		FILE *fp = fopen(fname.c_str(), "w"), *fp_dfde = fopen(fname_dfde.c_str(), "w");
		double sum_dfde = std::accumulate(dfde.begin(), dfde.end(), decltype(dfde)::value_type(0));
		fprintf(fp, "# energy (eV) Bin2\n"); fprintf(fp_dfde, "# energy (eV) dfde (sum = %14.7le)\n", sum_dfde);
		for (int ik = 0; ik < nk; ik++)
		for (int b = 0; b < nb_dm; b++){
			fprintf(fp, "%14.7le %14.7le\n", e_dm[ik][b] / eV, std::pow(dekn[ik*nb_dm + b] - de_mean, 2));
			fprintf(fp_dfde, "%14.7le %14.7le\n", e_dm[ik][b] / eV, dfde[ik*nb_dm + b]);
		}
		fclose(fp); fclose(fp_dfde);

		if (ionode){ printf("\nDone\n\n"); }
	}
}