#include "ElectronPhonon.h"

void electronphonon::analyse_g2(double de, double degauss, double degthr){
	int nfile_gm = last_file_index("ldbd_data/ldbd_gm.bin.", "") + 1,
		nfile_wq = last_file_index("ldbd_data/ldbd_wq_kpair.bin.", "") + 1;
	if (!exists("ldbd_data/ldbd_gm.bin.0") || mp->nprocs != nfile_gm || mp->nprocs != nfile_wq) return;
	if (ionode) printf("\n");
	if (ionode) printf("**************************************************\n");
	if (ionode) printf("analyse g2 for e-ph\n");
	if (ionode) printf("**************************************************\n");
	if (ionode && !is_dir("eph_analysis")) system("mkdir eph_analysis");

	// read electron energy range
	double ebot, etop;
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char stmp[200];
	for (int i = 0; i < 8; i++)
		fgets(stmp, sizeof stmp, fp);
	if (fgets(stmp, sizeof stmp, fp) != NULL){
		double dtmp1, dtmp2, dtmp3, dtmp4;
		sscanf(stmp, "%le %le %le %le %le %le", &dtmp1, &dtmp2, &dtmp3, &dtmp4, &ebot, &etop); if (ionode) printf("ebot = %14.7le eV etop = %14.7le eV\n", ebot / eV, etop / eV);
	}
	if (nv > 0 && elec->evmax <= ebot) error_message("evmax <= ebot", "analyse_g2");
	if (nb > nv && elec->ecmin >= etop) error_message("ecmin >= etop", "analyse_g2");
	fclose(fp);

	// transition energy grids
	double wstep = 0.001 * eV;
	int nw = round(std::min(ph->omega_max, etop - ebot) / wstep) + 1; // energy step 1 meV
	std::vector<double> wgrid(nw);
	for (int iw = 1; iw < nw; iw++)
		wgrid[iw] = wgrid[iw - 1] + wstep;

	// electron energy grids
	double gap = elec->ecmin - elec->evmax;
	int ne_v, ne_c, ne;
	std::vector<double> egrid;
	if (nv > 0 && nb > nv && gap <= 1.002*de){
		ne = ceil((etop - ebot) / de);
		egrid.resize(ne, 0);
		egrid[0] = ebot;
		for (int ie = 1; ie < ne; ie++)
			egrid[ie] = egrid[ie - 1] + de;
	}
	else{
		ne_v = nv > 0 ? ceil((elec->evmax - ebot) / de) + 1 : 0;
		ne_c = nb > nv ? ceil((etop - elec->ecmin) / de) + 1 : 0;
		ne = ne_v + ne_c;
		egrid.resize(ne, 0);
		if (ne_v > 0) egrid[ne_v - 1] = elec->evmax + 0.501 * de; // shift a little to ensure vbm is closer to egrid[ne_v - 2] than egrid[ne_v - 1]
		for (int ie = ne_v - 2; ie >= 0; ie--)
			egrid[ie] = egrid[ie + 1] - de;
		if (ne_c > 0) egrid[ne_v] = elec->ecmin - 0.501 * de; // shift a little to ensure cbm is closer to egrid[ne_v + 1] than egrid[ne_v]
		for (int ie = ne_v + 1; ie < ne; ie++)
			egrid[ie] = egrid[ie - 1] + de;
	}

	// q length grids
	double dql = ph->qmin;
	double qmax = ph->qmax;
	if (alg.only_intravalley || alg.only_intervalley || latt->vpos.size() >= 2){
		qmax = dql;
		for (size_t ik = 0; ik < nk_glob; ik++)
		for (size_t jk = ik + 1; jk < nk_glob; jk++){
			int iv1, iv2;
			if (!latt->kpair_is_allowed(elec->kvec[ik], elec->kvec[jk], iv1, iv2)) continue;
			double qlength = latt->qana(elec->kvec[ik], elec->kvec[jk], iv1, iv2);
			if (qlength > qmax){
				qmax = qlength;
				//if (ionode)
				//	printf("k1: %lg %lg %lg (%d)  k2: %lg %lg %lg (%d)  q= %lg\n",
				//		elec->kvec[ik][0], elec->kvec[ik][1], elec->kvec[ik][2], iv1,
				//		elec->kvec[jk][0], elec->kvec[jk][1], elec->kvec[jk][2], iv2, qmax);
			}
		}
	}

	int nql = ceil(qmax / dql) + 1;
	std::vector<double> qlgrid(nql);
	qlgrid[0] = 0;
	for (int iql = 1; iql < nql; iql++)
		qlgrid[iql] = qlgrid[iql - 1] + dql;

	// occupation
	double **f = alloc_real_array(nk_glob, nb);
	trunc_copy_array(f, elec->f_dm, nk_glob, bStart, bEnd);

	double sum_dfde = 0;
	for (int ik = 0; ik < nk_glob; ik++){
		for (int b = 0; b < nb; b++)
			sum_dfde += f[ik][b] * (1 - f[ik][b]);
	}

	//*****************************************************************
	// frequency-dependent spin-flip/conserving overlap square and number of spin-flip/conserving transitions are defined as
	// gsf2_{kn}(w) = (1/Nsf_{kn}(w)) (1/Nk) sum_{k'n'} |g_{kn,k'n'}|^2 step(spin-flip) delta(e_{kn}-e{k'n'}+-w) 
	// Nsf_{kn}(w) = (1/Nk) sum_{k'n'} step(spin-flip) delta(e_{kn}-e{k'n'}+-w)
	// gsc2_{kn}(w) = (1/Nsc_{kn}(w)) (1/Nk) sum_{k'n'} |g_{kn,k'n'}|^2 step(spin-conserving) delta(e_{kn}-e{k'n'}+-w)
	// Nsf_{kn}(w) = (1/Nk) sum_{k'n'} step(spin-flip) delta(e_{kn}-e{k'n'}+-w)
	//*****************************************************************

	double ***gsf2ew = alloc_real_array(3, ne, nw), ***Nsfew = alloc_real_array(3, ne, nw),
		***gsc2ew = alloc_real_array(3, ne, nw), ***Nscew = alloc_real_array(3, ne, nw);
	double **gsf2w_avg = alloc_real_array(3, nw), **Nsfw = alloc_real_array(3, nw),
		**gsc2w_avg = alloc_real_array(3, nw), **Nscw = alloc_real_array(3, nw);
	double **gsc2q = alloc_real_array(3, nql), **Nscq = alloc_real_array(3, nql),
		**gsf2q = alloc_real_array(3, nql), **Nsfq = alloc_real_array(3, nql);

	MPI_Barrier(MPI_COMM_WORLD);
	string fname_gm = "ldbd_data/ldbd_gm.bin." + int2str(mp->myrank),
		fname_wq = "ldbd_data/ldbd_wq_kpair.bin." + int2str(mp->myrank);
	if (ionode) printf("read %s and %s\n", fname_gm.c_str(), fname_wq.c_str());
	//printf("rank %d read %s and %s\n", mp->myrank, fname_gm.c_str(), fname_wq.c_str());
	FILE *fpgm = fopen(fname_gm.c_str(), "rb"), *fpwq = fopen(fname_wq.c_str(), "rb");
	/*
	size_t fgm_size = file_size(fpgm), fwq_size = file_size(fpwq);
	if (fgm_size % (16 * nm * nb * nb) != 0)
		error_message("fgm_size % (16 * nm * nb * nb) != 0", "electronphonon::analyse_g2");
	if (fwq_size % (8 * nm) != 0)
		error_message("fwq_size % (8 * nm) != 0", "electronphonon::analyse_g2");
	int n_index_g = fgm_size / (16 * nm * nb * nb), n_index_w = fwq_size / (8 * nm);
	if (n_index_g != n_index_w)
		error_message("n_index_g != n_index_w", "electronphonon::analyse_g2");
	printf("rank %d  n_index_g= %d\n", mp->myrank, n_index_g);
	complex ***g = alloc_array(n_index_g, nm, nb*nb);
	double **wq = alloc_real_array(n_index_g, nm);
	for (int i = 0; i < n_index_g; i++){
		printf("rank %d  i= %d\n", mp->myrank, i);
		fread(g[i][0], 2 * sizeof(double), nm*nb*nb, fpgm);
		fread(wq[i], sizeof(double), nm, fpwq);
	}
	fclose(fpgm); fclose(fpwq);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("rank %d  fsize= %lu  fsize= %lu\n", mp->myrank, fgm_size, fwq_size);
	MPI_Barrier(MPI_COMM_WORLD);
	*/
	//size_t expected_size = nkpair_proc*nb*nb * 2 * sizeof(double); // not right
	//check_file_size(fpgm, expected_size, fname_gm + " size does not match expected size");
	//expected_size = nkpair_proc*nm * sizeof(double); // not right
	//check_file_size(fpwq, expected_size, fname_wq + " size does not match expected size");

	bool isI[nk_glob];
	double sum_g2nq[nb*nb], **eig_sdeg = alloc_real_array(nk_glob, nb); // eigenvalues of energy-degeneracy projections of spin matrices
	complex g[nb*nb], mtmp[nb*nb], **U_sdeg = alloc_array(nk_glob, nb*nb);
	std::vector<double> nstates_e(ne);
	size_t count_nk_deg = 0;
	vector3<double> min_ds(1, 1, 1);
	std::vector<std::vector<int>> ie_ik(nk_glob, std::vector<int>(nb));

	double prefac_gaussexp = -0.5 / std::pow(degauss, 2);
	MPI_Barrier(MPI_COMM_WORLD);
	for (int id = 2; id >= 0; id--){
		// diagonalize spin matrices in energy-degenerate subspaces
		for (int ik = 0; ik < nk_glob; ik++){
			isI[ik] = diagonalize_deg(elec->s[ik][id], e[ik], nb, degthr, eig_sdeg[ik], U_sdeg[ik]);
			if (id == 2 && !isI[ik]) count_nk_deg++;

			// determine energy index of e[ik][b1] in egrid
			if (id == 2){ // run once
				for (int b1 = 0; b1 < nb; b1++){
					ie_ik[ik][b1] = b1 < nv ?
						round((e[ik][b1] - egrid[0]) / de) :
						round((e[ik][b1] - egrid[ne_v]) / de) + ne_v;
					int ie1 = ie_ik[ik][b1];
					if (ie1 >= 0 && ie1 < ne){
						nstates_e[ie1] += 1;
						if (fabs(e[ik][b1] - egrid[ie1]) > 0.501*de){
							printf("e[%d][%d]= %14.7le egrid[%d]= %14.7le 0.501*de= %14.7le\n", ik, b1, e[ik][b1], ie1, egrid[ie1], 0.501*de);
							error_message("|e[ik][b1] - egrid[ie1]| > 0.501*de", "analyse_g2");
						}
					}
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		rewind(fpgm); rewind(fpwq);
		int index_g = 0;
		for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
			int ik0 = k1st[ikpair_local];
			int jk0 = k2nd[ikpair_local];
			int iv1, iv2;
			//printf("rank %d  ik0= %d  jk0= %d  ikpair= %d (%d)\n", mp->myrank, ik0, jk0, ikpair_local, nkpair_proc);
			//printf("rank %d k1: %lg %lg %lg  k2: %lg %lg %lg\n", mp->myrank,
			//	elec->kvec[ik0][0], elec->kvec[ik0][1], elec->kvec[ik0][2],
			//	elec->kvec[jk0][0], elec->kvec[jk0][1], elec->kvec[jk0][2]);
			bool kpair_is_allowed = latt->kpair_is_allowed(elec->kvec[ik0], elec->kvec[jk0], iv1, iv2);
			//printf("rank %d kpair_is_allowed = %d\n", mp->myrank, kpair_is_allowed);
			double qlength = latt->qana(elec->kvec[ik0], elec->kvec[jk0], iv1, iv2);
			//double qlength = latt->klength(elec->kvec[ik0] - elec->kvec[jk0]);
			//printf("rank %d qlength = %lg\n", mp->myrank, qlength);
			int iql = round(qlength / dql);
			//printf("rank %d iql = %d dql = %lg\n", mp->myrank, iql, dql);
			if (kpair_is_allowed && (iql < 0 || iql >= nql)) error_message("iql < 0 || iql >= nql", "analyse_g2");
			//printf("rank %d k1: %lg %lg %lg (%d)  k2: %lg %lg %lg (%d)  q= %lg\n", mp->myrank,
			//	elec->kvec[ik0][0], elec->kvec[ik0][1], elec->kvec[ik0][2], iv1,
			//	elec->kvec[jk0][0], elec->kvec[jk0][1], elec->kvec[jk0][2], iv2, qlength);

			int nrun = (ik0 == jk0) ? 1 : 2;
			for (int irun = 0; irun < nrun; irun++){
				int ik, jk;
				if (irun == 0){ ik = ik0; jk = jk0; }
				else{ ik = jk0; jk = ik0; }
				index_g++;
				//printf("rank %d  ik= %d  jk= %d\n", mp->myrank, ik, jk);

				zeros(sum_g2nq, nb*nb);
				//printf("rank %d debug 1\n", mp->myrank);
				for (int im = 0; im < nm; im++){
					//printf("rank %d  im= %d  fpgm= %lu\n", mp->myrank, im, ftell(fpgm));
					if (fread(g, 2 * sizeof(double), nb*nb, fpgm) == nb*nb){}
					else { error_message("error during reading gm", "analyse_g2"); }
					//printf("rank %d  fpgm= %lu\n", mp->myrank, ftell(fpgm));

					double wq;
					//printf("rank %d  fpwq= %lu\n", mp->myrank, ftell(fpwq));
					if (fread(&wq, sizeof(double), 1, fpwq) == 1){}
					else { error_message("error during reading wq", "analyse_g2"); }
					//printf("rank %d  fpwq= %lu\n", mp->myrank, ftell(fpwq));

					if (!kpair_is_allowed) continue;
					//printf("rank %d  wq= %lg\n", mp->myrank, wq);
					if (im < ph->modeStart || im >= ph->modeEnd) continue;

					double nq = ph->bose(elec->temperature, wq);
					//double nq = ph->bose(elec->temperature, wq[index_g][im]);
					//printf("rank %d  nq= %lg\n", mp->myrank, nq);

					transpose(g, mtmp, nb); // from Fortran to C
					if (irun == 0) axbyc(g, mtmp, nb*nb);
					else hermite(mtmp, g, nb);
					//transpose(g[index_g][im], mtmp, nb); // from Fortran to C
					//if (irun == 0) axbyc(g[index_g][im], mtmp, nb*nb);
					//else hermite(mtmp, g[index_g][im], nb);
					//printf("rank %d  |g[0,0]|^2= %lg   |g[1,1]|^2= %lg\n", mp->myrank, g[index_g][im][0].norm(), g[index_g][im][nb + 1].norm());

					if (!isI[jk]) zgemm_interface(mtmp, g, U_sdeg[jk], nb);
					if (!isI[ik]) zgemm_interface(g, U_sdeg[ik], mtmp, nb, c1, c0, CblasConjTrans);
					//if (!isI[jk]) zgemm_interface(mtmp, g[index_g][im], U_sdeg[jk], nb);
					//if (!isI[ik]) zgemm_interface(g[index_g][im], U_sdeg[ik], mtmp, nb, c1, c0, CblasConjTrans);

					for (int b1 = 0; b1 < nb; b1++)
					for (int b2 = 0; b2 < nb; b2++)
						sum_g2nq[b1*nb + b2] += g[b1*nb + b2].norm() * nq;
						//sum_g2nq[b1*nb + b2] += g[index_g][im][b1*nb + b2].norm() * nq;
					//printf("rank %d  |sum_g2nq[0,0]|^2= %lg   |sum_g2nq[1,1]|^2= %lg\n", mp->myrank, sum_g2nq[0], sum_g2nq[nb + 1]);
				}

				if (!kpair_is_allowed) continue;
				//printf("rank %d debug 2\n", mp->myrank);

				for (int b1 = 0; b1 < nb; b1++){
					int ie1 = ie_ik[ik][b1];
					if (ie1 < 0 || ie1 >= ne) continue;

					for (int b2 = 0; b2 < nb; b2++){
						bool step_sf = eig_sdeg[ik][b1] * eig_sdeg[jk][b2] < 0; // true for a spin-flip transition
						if (step_sf && abs(eig_sdeg[ik][b1] - eig_sdeg[jk][b2]) < min_ds[id])
							min_ds[id] = abs(eig_sdeg[ik][b1] - eig_sdeg[jk][b2]);

						//printf("rank %d b1= %d b2= %d step_sf= %d\n", mp->myrank, b1, b2, step_sf);
						double f2f1bar = f[jk][b2] * (1 - f[ik][b1]);
						double f1f2bar = f[ik][b1] * (1 - f[jk][b2]);

						// transition energy w distribution
						double de = e[ik][b1] - e[jk][b2];
						for (int iw = 0; iw < nw; iw++){
							double delta_minus = exp(prefac_gaussexp * std::pow(de - wgrid[iw], 2)); // gaussian delta without prefactor
							double delta_plus = exp(prefac_gaussexp * std::pow(de + wgrid[iw], 2)); // gaussian delta without prefactor
							double weight_g2ew = delta_minus + delta_plus;
							double weight_g2w = f2f1bar * delta_minus + f1f2bar * delta_plus;

							if (step_sf){
								// frequency-dependent spin-flip overlap square and number of spin-flip transitions
								gsf2ew[id][ie1][iw] += sum_g2nq[b1*nb + b2] * weight_g2ew;
								Nsfew[id][ie1][iw] += weight_g2ew;
								gsf2w_avg[id][iw] += sum_g2nq[b1*nb + b2] * weight_g2w;
								Nsfw[id][iw] += weight_g2w;
							}
							else{
								// frequency-dependent spin-conserving overlap square and number of spin-conserving transitions
								gsc2ew[id][ie1][iw] += sum_g2nq[b1*nb + b2] * weight_g2ew;
								Nscew[id][ie1][iw] += weight_g2ew;
								gsc2w_avg[id][iw] += sum_g2nq[b1*nb + b2] * weight_g2w;
								Nscw[id][iw] += weight_g2w;
							}
						}

						// q-length distribution
						if (step_sf){
							gsf2q[id][iql] += sum_g2nq[b1*nb + b2];
							Nsfq[id][iql] += 1;
						}
						else{
							gsc2q[id][iql] += sum_g2nq[b1*nb + b2];
							Nscq[id][iql] += 1;
						}
					}
				}
				//printf("rank %d debug 3\n", mp->myrank);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		// collecting data
		mp->allreduce(gsf2ew[id], ne, nw, MPI_SUM); mp->allreduce(Nsfew[id], ne, nw, MPI_SUM);
		mp->allreduce(gsc2ew[id], ne, nw, MPI_SUM); mp->allreduce(Nscew[id], ne, nw, MPI_SUM);
		mp->allreduce(min_ds[id], MPI_MIN);
	}
	fclose(fpgm); fclose(fpwq);
	//dealloc_real_array(wq);
	//for (int i = 0; i < n_index_g; i++){ dealloc_array(g[i]); }
	// collecting data
	mp->allreduce(gsf2w_avg, 3, nw, MPI_SUM); mp->allreduce(Nsfw, 3, nw, MPI_SUM);
	mp->allreduce(gsc2w_avg, 3, nw, MPI_SUM); mp->allreduce(Nscw, 3, nw, MPI_SUM);
	mp->allreduce(gsf2q, 3, nql, MPI_SUM); mp->allreduce(Nsfq, 3, nql, MPI_SUM);
	mp->allreduce(gsc2q, 3, nql, MPI_SUM); mp->allreduce(Nscq, 3, nql, MPI_SUM);
	// useful information
	if (ionode){
		printf("no. of k with energy degeneracy (tot no. of k): %lu (%d)\n", count_nk_deg, nk_glob);
		printf("min. spin change along x: %lg\n", min_ds[0]);
		printf("min. spin change along y: %lg\n", min_ds[1]);
		printf("min. spin change along z: %lg\n", min_ds[2]); fflush(stdout);
	}
	// divide normalization factor
	double prefac_gauss = 1. / (sqrt(2 * M_PI) * degauss);
	double prefac_Nw = prefac_gauss / elec->nk_full / elec->nk_full;
	for (int id = 2; id >= 0; id--){
		for (int ie = 0; ie < ne; ie++){
			double prefac_New = prefac_gauss / elec->nk_full / nstates_e[ie];
			for (int iw = 0; iw < nw; iw++){
				gsf2ew[id][ie][iw] = gsf2ew[id][ie][iw] / Nsfew[id][ie][iw];
				gsc2ew[id][ie][iw] = gsc2ew[id][ie][iw] / Nscew[id][ie][iw];
				Nsfew[id][ie][iw] *= prefac_New;
				Nscew[id][ie][iw] *= prefac_New;
			}
		}
		for (int iw = 0; iw < nw; iw++){
			gsf2w_avg[id][iw] = gsf2w_avg[id][iw] / Nsfw[id][iw];
			gsc2w_avg[id][iw] = gsc2w_avg[id][iw] / Nscw[id][iw];
			Nsfw[id][iw] *= prefac_Nw;
			Nscw[id][iw] *= prefac_Nw;
		}
		for (int iql = 0; iql < nql; iql++){
			gsf2q[id][iql] = gsf2q[id][iql] / Nsfq[id][iql];
			gsc2q[id][iql] = gsc2q[id][iql] / Nscq[id][iql];
		}
	}

	// output frequency-dependent spin-flip/conserving overlap square and number of spin-flip/conserving transitions
	if (ionode){
		sum_dfde /= elec->nk_full;
		string sdir[3]; sdir[0] = "x"; sdir[1] = "y"; sdir[2] = "z";
		for (int id = 2; id >= 0; id--){
			string fnamesf = "eph_analysis/gsf2w_" + sdir[id] + ".out",
				fnamesc = "eph_analysis/gsc2w_" + sdir[id] + ".out";
			FILE *fpsf = fopen(fnamesf.c_str(), "w"),
				*fpsc = fopen(fnamesc.c_str(), "w");
			fprintf(fpsf, "# transition energy (meV) gsf2w Nsfw gsf2*Nsf/Nf\n");
			fprintf(fpsc, "# transition energy (meV) gsc2w Nscw gsc2*Nsc/Nf\n");
			for (int iw = 0; iw < nw; iw++){
				fprintf(fpsf, "%14.7le %14.7le %14.7le %14.7le\n", wgrid[iw] / eV * 1000, gsf2w_avg[id][iw], Nsfw[id][iw], gsf2w_avg[id][iw] * Nsfw[id][iw] / sum_dfde);
				fprintf(fpsc, "%14.7le %14.7le %14.7le %14.7le\n", wgrid[iw] / eV * 1000, gsc2w_avg[id][iw], Nscw[id][iw], gsc2w_avg[id][iw] * Nscw[id][iw] / sum_dfde);
			}
			fclose(fpsf); fclose(fpsc);

			for (int iw = 0; iw < nw; iw++){
				string fnamesf = "eph_analysis/gsf2ew_" + sdir[id] + "_w" + int2str(iw) + ".out",
					fnamesc = "eph_analysis/gsc2ew_" + sdir[id] + "_w" + int2str(iw) + ".out";
				FILE *fpsf = fopen(fnamesf.c_str(), "w"),
					*fpsc = fopen(fnamesc.c_str(), "w");
				fprintf(fpsf, "# elec. energy (eV) gsf2ew Nsfew gsf2e*Nsfe (for transition energy: %lg meV)\n", wgrid[iw] / eV * 1000);
				fprintf(fpsc, "# elec. energy (eV) gsc2ew Nscew gsc2e*Nsce (for transition energy: %lg meV)\n", wgrid[iw] / eV * 1000);
				for (int ie = 0; ie < ne; ie++){
					fprintf(fpsf, "%14.7le %14.7le %14.7le %14.7le\n", egrid[ie] / eV, gsf2ew[id][ie][iw], Nsfew[id][ie][iw], gsf2ew[id][ie][iw] * Nsfew[id][ie][iw]);
					fprintf(fpsc, "%14.7le %14.7le %14.7le %14.7le\n", egrid[ie] / eV, gsc2ew[id][ie][iw], Nscew[id][ie][iw], gsc2ew[id][ie][iw] * Nscew[id][ie][iw]);
				}
				fclose(fpsf); fclose(fpsc);
			}

			fnamesf = "eph_analysis/gsf2q_" + sdir[id] + ".out"; fnamesc = "eph_analysis/gsc2q_" + sdir[id] + ".out";
			fpsf = fopen(fnamesf.c_str(), "w"); fpsc = fopen(fnamesc.c_str(), "w");
			fprintf(fpsf, "# q length (bohr^-1) gsf2q Nsfq\n");
			fprintf(fpsc, "# q length (bohr^-1) gsc2q Nscq\n");
			for (int iql = 0; iql < nql; iql++){
				fprintf(fpsf, "%14.7le %14.7le %14.7le\n", qlgrid[iql], gsf2q[id][iql], Nsfq[id][iql]);
				fprintf(fpsc, "%14.7le %14.7le %14.7le\n", qlgrid[iql], gsc2q[id][iql], Nscq[id][iql]);
			}
			fclose(fpsf); fclose(fpsc);
		}

		// dos
		std::vector<double> dos(ne);
		for (int ie = 0; ie < ne; ie++)
			dos[ie] = nstates_e[ie] / de / elec->nk_full;
		double sum_dossq = 0, sum_dos = 0;
		for (int ie = 0; ie < ne; ie++){
			double fe = electron::fermi(elec->temperature, elec->mu, egrid[ie]);
			double dfde = fe * (1 - fe);
			sum_dossq += dfde * dos[ie] * dos[ie];
			sum_dos += dfde * dos[ie];
		}
		printf("effective scattering dos (inaccurate) = %14.7le\n", sum_dossq / sum_dos);
	}

	//deallocate memory
	dealloc_real_array(f);
	dealloc_real_array(eig_sdeg); dealloc_array(U_sdeg);
	dealloc_real_array(gsf2ew); dealloc_real_array(Nsfew); dealloc_real_array(gsf2w_avg); dealloc_real_array(Nsfw);
	dealloc_real_array(gsc2ew); dealloc_real_array(Nscew); dealloc_real_array(gsc2w_avg); dealloc_real_array(Nscw);
	dealloc_real_array(gsf2q); dealloc_real_array(Nsfq); dealloc_real_array(gsc2q); dealloc_real_array(Nscq);
}

void electronphonon::analyse_g2_ei(double de, double degauss, double degthr){
	int nfile_g = last_file_index("ldbd_data/ldbd_g.bin.", "") + 1;
	if (!exists("ldbd_data/ldbd_g.bin.0") || mp->nprocs != nfile_g) return;
	if (ionode) printf("\n");
	if (ionode) printf("**************************************************\n");
	if (ionode) printf("analyse g2 for e-i\n");
	if (ionode) printf("**************************************************\n");
	if (ionode && !is_dir("eimp_analysis")) system("mkdir eimp_analysis");

	// read electron energy range
	double ebot, etop;
	FILE *fp = fopen("ldbd_data/ldbd_size.dat", "r");
	char stmp[200];
	for (int i = 0; i < 8; i++)
		fgets(stmp, sizeof stmp, fp);
	if (fgets(stmp, sizeof stmp, fp) != NULL){
		double dtmp1, dtmp2, dtmp3, dtmp4;
		sscanf(stmp, "%le %le %le %le %le %le", &dtmp1, &dtmp2, &dtmp3, &dtmp4, &ebot, &etop); if (ionode) printf("ebot = %14.7le eV etop = %14.7le eV\n", ebot / eV, etop / eV);
	}
	if (nv > 0 && elec->evmax <= ebot) error_message("evmax <= ebot", "eimp::analyse_g2");
	if (nb > nv && elec->ecmin >= etop) error_message("ecmin >= etop", "eimp::analyse_g2");
	fclose(fp);

	// electron energy grids
	double gap = elec->ecmin - elec->evmax;
	int ne_v, ne_c, ne;
	std::vector<double> egrid;
	if (nv > 0 && nb > nv && gap <= 1.002*de){
		ne = ceil((etop - ebot) / de);
		egrid.resize(ne, 0);
		egrid[0] = ebot;
		for (int ie = 1; ie < ne; ie++)
			egrid[ie] = egrid[ie - 1] + de;
	}
	else{
		ne_v = nv > 0 ? ceil((elec->evmax - ebot) / de) + 1 : 0;
		ne_c = nb > nv ? ceil((etop - elec->ecmin) / de) + 1 : 0;
		ne = ne_v + ne_c;
		egrid.resize(ne, 0);
		if (ne_v > 0) egrid[ne_v - 1] = elec->evmax + 0.501 * de; // shift a little to ensure vbm is closer to egrid[ne_v - 2] than egrid[ne_v - 1]
		for (int ie = ne_v - 2; ie >= 0; ie--)
			egrid[ie] = egrid[ie + 1] - de;
		if (ne_c > 0) egrid[ne_v] = elec->ecmin - 0.501 * de; // shift a little to ensure cbm is closer to egrid[ne_v + 1] than egrid[ne_v]
		for (int ie = ne_v + 1; ie < ne; ie++)
			egrid[ie] = egrid[ie - 1] + de;
	}

	// q length grids
	double dql = ph->qmin;
	int nql = ceil(ph->qmax / dql) + 1;
	std::vector<double> qlgrid(nql);
	qlgrid[0] = 0;
	for (int iql = 1; iql < nql; iql++)
		qlgrid[iql] = qlgrid[iql - 1] + dql;

	// occupation
	double **f = alloc_real_array(nk_glob, nb);
	trunc_copy_array(f, elec->f_dm, nk_glob, bStart, bEnd);

	double sum_dfde = 0;
	for (int ik = 0; ik < nk_glob; ik++){
		for (int b = 0; b < nb; b++)
			sum_dfde += f[ik][b] * (1 - f[ik][b]);
	}

	//*****************************************************************
	// spin-flip/conserving overlap square and number of spin-flip/conserving transitions are defined as
	// gsf2_{kn} = (1/Nsf_{kn}(w)) (1/Nk) sum_{k'n'} |g_{kn,k'n'}|^2 step(spin-flip) delta(e_{kn}-e{k'n'}) 
	// Nsf_{kn} = (1/Nk) sum_{k'n'} step(spin-flip) delta(e_{kn}-e{k'n'})
	// gsc2_{kn} = (1/Nsc_{kn}(w)) (1/Nk) sum_{k'n'} |g_{kn,k'n'}|^2 step(spin-conserving) delta(e_{kn}-e{k'n'})
	// Nsf_{kn} = (1/Nk) sum_{k'n'} step(spin-flip) delta(e_{kn}-e{k'n'})
	//*****************************************************************

	double **gsf2e = alloc_real_array(3, ne), **Nsfe = alloc_real_array(3, ne),
		**gsc2e = alloc_real_array(3, ne), **Nsce = alloc_real_array(3, ne);
	double gsf2_avg[3]{0}, Nsf[3]{0}, gsc2_avg[3]{0}, Nsc[3]{0};
	double **gsc2q = alloc_real_array(3, nql), **Nscq = alloc_real_array(3, nql),
		**gsf2q = alloc_real_array(3, nql), **Nsfq = alloc_real_array(3, nql);

	string fname_g = "ldbd_data/ldbd_g.bin." + int2str(mp->myrank);
	if (ionode) printf("read %s\n", fname_g.c_str());
	FILE *fpg = fopen(fname_g.c_str(), "rb");
	//size_t expected_size = nkpair_proc*nb*nb * 2 * sizeof(double); // not right
	//check_file_size(fpg, expected_size, fname_g + " size does not match expected size");

	bool isI[nk_glob];
	double **eig_sdeg = alloc_real_array(nk_glob, nb); // eigenvalues of energy-degeneracy projections of spin matrices
	complex g[nb*nb], mtmp[nb*nb], **U_sdeg = alloc_array(nk_glob, nb*nb);
	std::vector<double> nstates_e(ne);
	size_t count_nk_deg = 0;
	vector3<double> min_ds(1, 1, 1);
	std::vector<std::vector<int>> ie_ik(nk_glob, std::vector<int>(nb));

	double prefac_gaussexp = -0.5 / std::pow(degauss, 2);
	for (int id = 2; id >= 0; id--){
		// diagonalize spin matrices in energy-degenerate subspaces
		for (int ik = 0; ik < nk_glob; ik++){
			isI[ik] = diagonalize_deg(elec->s[ik][id], e[ik], nb, degthr, eig_sdeg[ik], U_sdeg[ik]);
			if (id == 2 && !isI[ik]) count_nk_deg++;

			// determine energy index of e[ik][b1] in egrid
			if (id == 2){ // run once
				for (int b1 = 0; b1 < nb; b1++){
					ie_ik[ik][b1] = b1 < nv ?
						round((e[ik][b1] - egrid[0]) / de) :
						round((e[ik][b1] - egrid[ne_v]) / de) + ne_v;
					int ie1 = ie_ik[ik][b1];
					if (ie1 >= 0 && ie1 < ne){
						nstates_e[ie1] += 1;
						if (fabs(e[ik][b1] - egrid[ie1]) > 0.501*de){
							printf("e[%d][%d]= %14.7le egrid[%d]= %14.7le 0.501*de= %14.7le\n", ik, b1, e[ik][b1], ie1, egrid[ie1], 0.501*de);
							error_message("|e[ik][b1] - egrid[ie1]| > 0.501*de", "analyse_g2");
						}
					}
				}
			}
		}

		rewind(fpg);
		for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
			int ik0 = k1st[ikpair_local];
			int jk0 = k2nd[ikpair_local];
			double qlength = latt->klength(elec->kvec[ik0] - elec->kvec[jk0]);
			int iql = round(qlength / dql);
			if (iql < 0 || iql >= nql) error_message("iql < 0 || iql >= nql", "eimp::analyse_g2");

			int nrun = (ik0 == jk0) ? 1 : 2;
			for (int irun = 0; irun < nrun; irun++){
				int ik, jk;
				if (irun == 0){ ik = ik0; jk = jk0; }
				else{ ik = jk0; jk = ik0; }

				if (fread(g, 2 * sizeof(double), nb*nb, fpg) == nb*nb){}
				else { error_message("error during reading g", "eimp::analyse_g2"); }
				transpose(g, mtmp, nb); // from Fortran to C

				if (irun == 0) axbyc(g, mtmp, nb*nb);
				else hermite(mtmp, g, nb);

				if (!isI[jk]) zgemm_interface(mtmp, g, U_sdeg[jk], nb);
				if (!isI[ik]) zgemm_interface(g, U_sdeg[ik], mtmp, nb, c1, c0, CblasConjTrans);

				for (int b1 = 0; b1 < nb; b1++){
					int ie1 = ie_ik[ik][b1];
					if (ie1 < 0 || ie1 >= ne) continue;

					for (int b2 = 0; b2 < nb; b2++){
						bool step_sf = eig_sdeg[ik][b1] * eig_sdeg[jk][b2] < 0; // true for a spin-flip transition
						if (step_sf && abs(eig_sdeg[ik][b1] - eig_sdeg[jk][b2]) < min_ds[id])
							min_ds[id] = abs(eig_sdeg[ik][b1] - eig_sdeg[jk][b2]);

						double g2_pair = g[b1*nb + b2].norm();

						double f2f1bar = f[jk][b2] * (1 - f[ik][b1]);
						double f1f2bar = f[ik][b1] * (1 - f[jk][b2]);

						// transition energy w distribution
						double de = e[ik][b1] - e[jk][b2];
						double delta = exp(prefac_gaussexp * std::pow(de, 2)); // gaussian delta without prefactor
						double weight_g2e = delta;
						double weight_g2 = 0.5 * (f2f1bar + f1f2bar) * delta;

						if (step_sf){
							// frequency-dependent spin-flip overlap square and number of spin-flip transitions
							gsf2e[id][ie1] += g2_pair * weight_g2e;
							Nsfe[id][ie1] += weight_g2e;
							gsf2_avg[id] += g2_pair * weight_g2;
							Nsf[id] += weight_g2;
						}
						else{
							// frequency-dependent spin-conserving overlap square and number of spin-conserving transitions
							gsc2e[id][ie1] += g2_pair * weight_g2e;
							Nsce[id][ie1] += weight_g2e;
							gsc2_avg[id] += g2_pair * weight_g2;
							Nsc[id] += weight_g2;
						}

						// q-length distribution
						if (step_sf){
							gsf2q[id][iql] += g2_pair;
							Nsfq[id][iql] += 1;
						}
						else{
							gsc2q[id][iql] += g2_pair;
							Nscq[id][iql] += 1;
						}
					}
				}
			}
		}
		// collecting data
		mp->allreduce(gsf2e[id], ne, MPI_SUM); mp->allreduce(Nsfe[id], ne, MPI_SUM);
		mp->allreduce(gsc2e[id], ne, MPI_SUM); mp->allreduce(Nsce[id], ne, MPI_SUM);
		mp->allreduce(min_ds[id], MPI_MIN);
	}
	fclose(fpg);
	// collecting data
	mp->allreduce(gsf2_avg, 3, MPI_SUM); mp->allreduce(Nsf, 3, MPI_SUM);
	mp->allreduce(gsc2_avg, 3, MPI_SUM); mp->allreduce(Nsc, 3, MPI_SUM);
	mp->allreduce(gsf2q, 3, nql, MPI_SUM); mp->allreduce(Nsfq, 3, nql, MPI_SUM);
	mp->allreduce(gsc2q, 3, nql, MPI_SUM); mp->allreduce(Nscq, 3, nql, MPI_SUM);
	// useful information
	if (ionode){
		printf("no. of k with energy degeneracy (tot no. of k): %lu (%d)\n", count_nk_deg, nk_glob);
		printf("min. spin change along x: %lg\n", min_ds[0]);
		printf("min. spin change along y: %lg\n", min_ds[1]);
		printf("min. spin change along z: %lg\n", min_ds[2]); fflush(stdout);
	}
	// divide normalization factor
	double prefac_gauss = 1. / (sqrt(2 * M_PI) * degauss);
	double prefac_N = prefac_gauss / elec->nk_full / elec->nk_full;
	for (int id = 2; id >= 0; id--){
		for (int ie = 0; ie < ne; ie++){
			double prefac_Ne = prefac_gauss / elec->nk_full / nstates_e[ie];
			gsf2e[id][ie] = gsf2e[id][ie] / Nsfe[id][ie];
			gsc2e[id][ie] = gsc2e[id][ie] / Nsce[id][ie];
			Nsfe[id][ie] *= prefac_Ne;
			Nsce[id][ie] *= prefac_Ne;
		}
		gsf2_avg[id] = gsf2_avg[id] / Nsf[id];
		gsc2_avg[id] = gsc2_avg[id] / Nsc[id];
		Nsf[id] *= prefac_N;
		Nsc[id] *= prefac_N;
		for (int iql = 0; iql < nql; iql++){
			gsf2q[id][iql] = gsf2q[id][iql] / Nsfq[id][iql];
			gsc2q[id][iql] = gsc2q[id][iql] / Nscq[id][iql];
		}
	}

	// output spin-flip/conserving overlap square and number of spin-flip/conserving transitions
	if (ionode){
		sum_dfde /= elec->nk_full;
		string sdir[3]; sdir[0] = "x"; sdir[1] = "y"; sdir[2] = "z";

		string fnamesf = "eimp_analysis/gsf2.out",
			fnamesc = "eimp_analysis/gsc2.out";
		FILE *fpsf = fopen(fnamesf.c_str(), "w"),
			*fpsc = fopen(fnamesc.c_str(), "w");
		fprintf(fpsf, "# dir gsf2w Nsfw gsf2*Nsf/Nf\n");
		fprintf(fpsc, "# dir gsc2w Nscw gsc2*Nsc/Nf\n");
		for (int id = 2; id >= 0; id--){
			fprintf(fpsf, "%s %14.7le %14.7le %14.7le\n", sdir[id].c_str(), gsf2_avg[id], Nsf[id], gsf2_avg[id] * Nsf[id] / sum_dfde);
			fprintf(fpsc, "%s %14.7le %14.7le %14.7le\n", sdir[id].c_str(), gsc2_avg[id], Nsc[id], gsc2_avg[id] * Nsc[id] / sum_dfde);
		}
		fclose(fpsf); fclose(fpsc);

		for (int id = 2; id >= 0; id--){
			string fnamesf = "eimp_analysis/gsf2e_" + sdir[id] + ".out",
				fnamesc = "eimp_analysis/gsc2e_" + sdir[id] + ".out";
			FILE *fpsf = fopen(fnamesf.c_str(), "w"),
				*fpsc = fopen(fnamesc.c_str(), "w");
			fprintf(fpsf, "# elec. energy (eV) gsf2e Nsfe gsf2e*Nsfe\n");
			fprintf(fpsc, "# elec. energy (eV) gsc2e Nsce gsc2e*Nsce\n");
			for (int ie = 0; ie < ne; ie++){
				fprintf(fpsf, "%14.7le %14.7le %14.7le %14.7le\n", egrid[ie] / eV, gsf2e[id][ie], Nsfe[id][ie], gsf2e[id][ie] * Nsfe[id][ie]);
				fprintf(fpsc, "%14.7le %14.7le %14.7le %14.7le\n", egrid[ie] / eV, gsc2e[id][ie], Nsce[id][ie], gsc2e[id][ie] * Nsce[id][ie]);
			}
			fclose(fpsf); fclose(fpsc);

			fnamesf = "eimp_analysis/gsf2q_" + sdir[id] + ".out"; fnamesc = "eimp_analysis/gsc2q_" + sdir[id] + ".out";
			fpsf = fopen(fnamesf.c_str(), "w"); fpsc = fopen(fnamesc.c_str(), "w");
			fprintf(fpsf, "# q length (bohr^-1) gsf2q Nsfq\n");
			fprintf(fpsc, "# q length (bohr^-1) gsc2q Nscq\n");
			for (int iql = 0; iql < nql; iql++){
				fprintf(fpsf, "%14.7le %14.7le %14.7le\n", qlgrid[iql], gsf2q[id][iql], Nsfq[id][iql]);
				fprintf(fpsc, "%14.7le %14.7le %14.7le\n", qlgrid[iql], gsc2q[id][iql], Nscq[id][iql]);
			}
			fclose(fpsf); fclose(fpsc);
		}

		// dos
		std::vector<double> dos(ne);
		for (int ie = 0; ie < ne; ie++)
			dos[ie] = nstates_e[ie] / de / elec->nk_full;
		double sum_dossq = 0, sum_dos = 0;
		for (int ie = 0; ie < ne; ie++){
			double fe = electron::fermi(elec->temperature, elec->mu, egrid[ie]);
			double dfde = fe * (1 - fe);
			sum_dossq += dfde * dos[ie] * dos[ie];
			sum_dos += dfde * dos[ie];
		}
		printf("dos_avg = %14.7le\n", sum_dossq / sum_dos);
	}

	//deallocate memory
	dealloc_real_array(f);
	dealloc_real_array(eig_sdeg); dealloc_array(U_sdeg);
	dealloc_real_array(gsf2e); dealloc_real_array(Nsfe); dealloc_real_array(gsc2e); dealloc_real_array(Nsce);
	dealloc_real_array(gsf2q); dealloc_real_array(Nsfq); dealloc_real_array(gsc2q); dealloc_real_array(Nscq);
}

void electronphonon::compute_imsig(){
	if (!alg.scatt_enable || !alg.linearize) return;
	if (!alg.Pin_is_sparse) P1 = alloc_array(nkpair_proc, (int)std::pow(nb, 4));
	if (!alg.Pin_is_sparse) P2 = alloc_array(nkpair_proc, (int)std::pow(nb, 4));

	set_ephmat();
	compute_imsig("eph");
	for (int iD = 0; iD < eip.ni.size(); iD++){ if (!alg.only_ee) add_scatt_contrib("eimp", iD); }
	if (eip.ni.size() >0 && !alg.only_ee) compute_imsig("eph_eimp");
	//if (ee_model != nullptr && eep.eeMode == "Pee_fixed_at_eq" && !alg.only_eimp) { add_scatt_contrib("ee"); compute_imsig("eph_eimp_ee"); }
	if (ee_model != nullptr && !alg.only_eimp) { add_scatt_contrib("ee"); compute_imsig("eph_eimp_ee"); } // to have ImSigma and mobility due to e-e, this should be always called
	
	if (!alg.Pin_is_sparse) { dealloc_array(P1); dealloc_array(P2); }
	if (sP1 != nullptr) { delete sP1; sP1 = nullptr; }
	if (sP2 != nullptr) { delete sP2; sP2 = nullptr; }
}

void electronphonon::compute_imsig(string what){
	if (code != "jdftx") return;
	//if (alg.linearize){
	//	if (ionode) printf("\ncompute_imsig does not work with alg.linearize currently:\n");
	//	return;
	//}
	if (ionode) printf("\n");
	if (ionode) printf("**************************************************\n");
	if (ionode) printf("Compute carrier relaxation and transport properties for %s scattering:\n", what.c_str());
	if (ionode) printf("Please ensure the system is treated as a semiconductor, otherwise calculated mobility is meaningless:\n");
	if (ionode) printf("**************************************************\n");

	double prefac_imsig = M_PI / elec->nk_full;
	double **imsig = alloc_real_array(nk_glob, nb);
	double **f = trunc_alloccopy_array(elec->f_dm, nk_glob, bStart, bEnd);

	bool has_v = exists("ldbd_data/ldbd_vvec.bin"); double v0[3]{0};
	double ***v;
	double **imsigp = has_v ? alloc_real_array(nk_glob, nb) : nullptr;
	if (has_v){
		if (ionode) printf("read ldbd_data/ldbd_vvec.bin\n");
		v = alloc_real_array(nk_glob, nb, 3);
		FILE *fpv = fopen("ldbd_data/ldbd_vvec.bin", "rb");
		size_t expected_size = nk_glob * nb * 3 * sizeof(double);
		check_file_size(fpv, expected_size, "ldbd_vvec.bin size does not match expected size");
		for (int ik = 0; ik < nk_glob; ik++)
			fread(v[ik][0], sizeof(double), nb * 3, fpv);
		fclose(fpv);
	}

	// state-resolved ImSigma and ImSigmaP
	complex *P1_, *P2_;
	if (sP1 != nullptr){
		P1_ = new complex[(int)std::pow(nb, 4)];
		P2_ = new complex[(int)std::pow(nb, 4)];
	}

	for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
		int ik_glob = k1st[ikpair_local];
		int ikp_glob = k2nd[ikpair_local];

		int iv1, iv2;
		if (!latt->kpair_is_allowed(elec->kvec[ik_glob], elec->kvec[ikp_glob], iv1, iv2)) continue;

		if (P1 != nullptr && sP1 == nullptr){ P1_ = P1[ikpair_local]; P2_ = P2[ikpair_local]; }
		else if (sP1 != nullptr){
			sP1->smat[ikpair_local]->todense(P1_, nb*nb, nb*nb); sP2->smat[ikpair_local]->todense(P2_, nb*nb, nb*nb);
		}

		for (int i1 = 0; i1 < nb; i1++){
			int i11 = i1*nb + i1;
			int n11 = i11*nb*nb;

			double *v1kn = has_v ? v[ik_glob][i1] : v0;
			vector3<> v1(v1kn[0], v1kn[1], v1kn[2]);

			for (int i2 = 0; i2 < nb; i2++){
				if (ik_glob == ikp_glob && i1 == i2) continue;

				int i22 = i2*nb + i2;
				int n22 = i22*nb*nb;

				double *v2kn = has_v ? v[ikp_glob][i2] : v0;
				vector3<> v2(v2kn[0], v2kn[1], v2kn[2]);

				double cosThetaScatter = dot(v1, v2) / sqrt(std::max(1e-16, v1.length_squared() * v2.length_squared()));

				if (e[ik_glob][i1] >= eStart && e[ik_glob][i1] <= eEnd){
					double imsig_12 = P1_[n11 + i22].real() * f[ikp_glob][i2] + (1 - f[ikp_glob][i2]) * P2_[n22 + i11].real();
					imsig[ik_glob][i1] += imsig_12; if (has_v) imsigp[ik_glob][i1] += imsig_12 * (1. - cosThetaScatter);
				}
				if (ik_glob < ikp_glob && e[ikp_glob][i2] >= eStart && e[ikp_glob][i2] <= eEnd){
					double imsig_21 = P2_[n22 + i11].real() * f[ik_glob][i1] + (1 - f[ik_glob][i1]) * P1_[n11 + i22].real();
					imsig[ikp_glob][i2] += imsig_21; if (has_v) imsigp[ikp_glob][i2] += imsig_21 * (1. - cosThetaScatter);
				}
			}
		}
	}
	if (not has_v and sP1 != nullptr){ delete[] P1_; delete[] P2_; P1_ = nullptr; P2_ = nullptr; }
	mp->allreduce(imsig, nk_glob, nb, MPI_SUM);
	if (has_v) mp->allreduce(imsigp, nk_glob, nb, MPI_SUM);

	axbyc(imsig, nullptr, nk_glob, nb, 0, prefac_imsig);
	if (has_v) axbyc(imsigp, nullptr, nk_glob, nb, 0, prefac_imsig);

	// write state-resolved quantities
	if (ionode){
		string fname = "imsig_" + what + ".out";
		FILE *fpimsig = fopen(fname.c_str(), "w");
		if (!has_v) fprintf(fpimsig, "#E -dF/dE ImSigma\n");
		else{
			fprintf(fpimsig, "#electron / hole density = %14.7le / %14.7le\n", elec->ne, elec->nh);
			fprintf(fpimsig, "#Nk = %lg Volume = %14.7le\n", elec->nk_full, latt->cell_size);
			fprintf(fpimsig, "#cm2byVs2au = %14.7le\n", cm2byVs2au);
			fprintf(fpimsig, "#E -dF/dE ImSigma ImSigmaP vx2 vy2 vz2\n");
		}
		for (int ik = 0; ik < nk_glob; ik++)
		for (int b = 0; b < nb; b++){
			double dfde = f[ik][b] * (1 - f[ik][b]) / elec->temperature;
			//if (!has_v && imsig[ik][b] > 1e-40) fprintf(fpimsig, "%14.7le %14.7le %14.7le\n", e[ik][b], dfde, imsig[ik][b]);
			//else if (has_v && imsigp[ik][b] > 1e-40 && imsig[ik][b] > 1e-40) 
			if (!has_v) fprintf(fpimsig, "%14.7le %14.7le %14.7le\n", e[ik][b], dfde, imsig[ik][b]);
			else
				fprintf(fpimsig, "%14.7le %14.7le %14.7le %14.7le %14.7le %14.7le %14.7le\n",
				e[ik][b], dfde, imsig[ik][b], imsigp[ik][b], std::pow(v[ik][b][0], 2), std::pow(v[ik][b][1], 2), std::pow(v[ik][b][2], 2));
		}
		fclose(fpimsig);
	}

	// average ImSigma and carrier lifetime
	double imsig_avg = elec->average_dfde(imsig, f, nk_glob, nb, false);
	if (ionode) printf("imsig_avg = %lg Ha = %lg meV\n", imsig_avg, imsig_avg / eV * 1000);
	if (ionode) printf("carrier lifetime (rate avg.) = %lg fs\n", 0.5 / imsig_avg / fs);
	double taup_avg = 0.5 * elec->average_dfde(imsig, f, nk_glob, nb, true);
	if (ionode) printf("carrier lifetime (time avg.) = %lg fs\n", taup_avg / fs);

	if (has_v){
		// average ImSigmaP and momentum lifetime (Eq. S10 and S11 of Nano Lett. 21, 9594 (2021))
		double imsigp_avg = elec->average_dfde(imsigp, f, nk_glob, nb, false);
		if (ionode) printf("imsigp_avg = %lg Ha = %lg meV\n", imsigp_avg, imsigp_avg / eV * 1000);
		if (ionode) printf("momentum lifetime (rate avg.) = %lg fs\n", 0.5 / imsigp_avg / fs);
		double taum_avg = 0.5 * elec->average_dfde(imsigp, f, nk_glob, nb, true);
		if (ionode) printf("momentum lifetime (time avg.) = %lg fs\n", taum_avg / fs);

		// mobility (Eq. S8 of Nano Lett. 21, 9594 (2021))
		matrix3<> cond_e(matrix3<>(0, 0, 0)), cond_h(matrix3<>(0, 0, 0)), cond(matrix3<>(0, 0, 0));
		if (nb > nv) cond_e = compute_conductivity_brange(imsigp, v, f, nv, nb);
		if (nv > 0) cond_h = compute_conductivity_brange(imsigp, v, f, 0, nv);
		cond = cond_e + cond_h;
		write_conductivity(cond);
		if (nb > nv) matrix3<> mob_e = compute_mobility_brange(cond_e, f, nv, nb, "electrons");
		if (nv > 0) matrix3<> mob_h = compute_mobility_brange(cond_h, f, 0, nv, "holes");
	}

	if (ionode) { printf("\n"); fflush(stdout); } MPI_Barrier(MPI_COMM_WORLD);
	if (not has_v){
		dealloc_real_array(f);
		dealloc_real_array(imsig);
		return;
	}

	//Iterative solution of linearized Boltzmann transport equation
	//(-df / dEfield)_1 = -v_1 dfde_1 tau_1 - tau_1 tau^(-1)_{21} (df / dEfield)_2
	//tau^(-1)_{21} = P_{2211} f_1 + P_{1122} (1-f_1)
	//Initial (-df / dEfield) = -v dfde tau_m
	//So define x = (-df / dEfield)_1, A = I - Ap, Ap = tau_1 tau^(-1)_{21}, b = -v_1 dfde_1 tau_1,
	//we have A x = b or the iterative equation x^{n+1} = Ap x^{n} + b
	if (ionode) { printf("\nIterative solution of linearized Boltzmann transport equation:\n"); fflush(stdout); }
	double prefac_lbte = 2*M_PI / elec->nk_full, **tau_p = alloc_real_array(nk_glob, nb);
	double ***B = alloc_real_array(nk_glob, nb, 3), ***dfdEfield = alloc_real_array(nk_glob, nb, 3);
	for (int ik = 0; ik < nk_glob; ik++)
	for (int b = 0; b < nb; b++){
		double dfde = f[ik][b] * (1 - f[ik][b]) / elec->temperature;
		double tau_m = imsigp[ik][b] > 1e-40 ? 0.5 / imsigp[ik][b] : 0;
		tau_p[ik][b] = imsigp[ik][b] > 1e-40 ? 0.5 / imsig[ik][b] : 0;
		for (int idir = 0; idir < 3; idir++){
			B[ik][b][idir] = v[ik][b][idir] * dfde * tau_m;
			dfdEfield[ik][b][idir] = v[ik][b][idir] * dfde * tau_m; //note that dfde is actually -df/de and dfdEfield is -df/dEfield
		}
	}

	matrix3<> cond_e(matrix3<>(0, 0, 0)), cond_h(matrix3<>(0, 0, 0)), cond(matrix3<>(0, 0, 0));
	if (nb > nv) cond_e += compute_conductivity_brange(dfdEfield, v, nv, nb);
	if (nv > 0) cond_h += compute_conductivity_brange(dfdEfield, v, 0, nv);
	cond = cond_e + cond_h;
	//if (ionode) { printf("Initial results:\n"); fflush(stdout); }
	//write_conductivity(cond);
	matrix3<> mob_e(matrix3<>(0, 0, 0)), mob_h(matrix3<>(0, 0, 0));
	if (nb > nv) mob_e = compute_mobility_brange(cond_e, f, nv, nb, "electrons", false);
	if (nv > 0) mob_h = compute_mobility_brange(cond_h, f, 0, nv, "holes", false);
	double sum_mob_abs = 0;
	for (int idir = 0; idir < 3; idir++)
	for (int jdir = 0; jdir < 3; jdir++)
		sum_mob_abs += fabs(mob_e(idir, jdir)) + fabs(mob_h(idir, jdir));

	double ***ApX = alloc_real_array(nk_glob, nb, 3);
	int iter = 0;
	while (iter < 1000){

		zeros(ApX, nk_glob, nb, 3);
		for (int ikpair_local = 0; ikpair_local < nkpair_proc; ikpair_local++){
			int ik_glob = k1st[ikpair_local];
			int ikp_glob = k2nd[ikpair_local];

			int iv1, iv2;
			if (!latt->kpair_is_allowed(elec->kvec[ik_glob], elec->kvec[ikp_glob], iv1, iv2)) continue;

			if (P1 != nullptr && sP1 == nullptr){ P1_ = P1[ikpair_local]; P2_ = P2[ikpair_local]; }
			else if (sP1 != nullptr){
				sP1->smat[ikpair_local]->todense(P1_, nb*nb, nb*nb); sP2->smat[ikpair_local]->todense(P2_, nb*nb, nb*nb);
			}

			for (int i1 = 0; i1 < nb; i1++){
				int i11 = i1*nb + i1;
				int n11 = i11*nb*nb;
				double f1 = f[ik_glob][i1], taup1 = tau_p[ik_glob][i1], *x1 = dfdEfield[ik_glob][i1];

				for (int i2 = 0; i2 < nb; i2++){
					if (ik_glob == ikp_glob && i1 == i2) continue;

					int i22 = i2*nb + i2;
					int n22 = i22*nb*nb;
					double f2 = f[ikp_glob][i2], taup2 = tau_p[ikp_glob][i2], *x2 = dfdEfield[ikp_glob][i2];

					double P1122 = P1_[n11 + i22].real(), P2211 = P2_[n22 + i11].real();

					//tau^(-1)_{21} = P_{2211} f_1 + P_{1122} (1-f_1)
					//x = (-df / dEfield)_1, A = I - Ap, Ap = tau_1 tau^(-1)_{21}, b = -v_1 dfde_1 tau_1,
					//we have A x = b or the iterative equation x^{n+1} = Ap x^{n} + b
					if (e[ik_glob][i1] >= eStart && e[ik_glob][i1] <= eEnd){
						double tauinv_12 = P1122 * f2 + (1 - f2) * P2211;
						for (int idir = 0; idir < 3; idir++)
							ApX[ikp_glob][i2][idir] += taup2 * tauinv_12 * x1[idir];
					}
					if (ik_glob < ikp_glob && e[ikp_glob][i2] >= eStart && e[ikp_glob][i2] <= eEnd){
						double tauinv_21 = P2211 * f1 + (1 - f1) * P1122;
						for (int idir = 0; idir < 3; idir++)
							ApX[ik_glob][i1][idir] += taup1 * tauinv_21 * x2[idir];
					}
				}
			}
		}
		mp->allreduce(ApX, nk_glob, nb, 3, MPI_SUM);

		//x^{ n + 1 } = Ap x^{ n } +b
		axbyc(dfdEfield, B, nk_glob, nb, 3);
		axbyc(dfdEfield, ApX, nk_glob, nb, 3, prefac_lbte, 1);

		//compute conductivity and mobility
		matrix3<> cond_e_new(matrix3<>(0, 0, 0)), cond_h_new(matrix3<>(0, 0, 0)), cond_new(matrix3<>(0, 0, 0));
		if (nb > nv) cond_e_new += compute_conductivity_brange(dfdEfield, v, nv, nb);
		if (nv > 0) cond_h_new += compute_conductivity_brange(dfdEfield, v, 0, nv);
		cond_new = cond_e_new + cond_h_new;
		matrix3<> mob_e_new(matrix3<>(0, 0, 0)), mob_h_new(matrix3<>(0, 0, 0));
		if (nb > nv) mob_e_new = compute_mobility_brange(cond_e_new, f, nv, nb, "electrons", false);
		if (nv > 0) mob_h_new = compute_mobility_brange(cond_h_new, f, 0, nv, "holes", false);

		//check if mobilities are converged
		double error = 0, rel_error = 0, sum_mob_abs_new = 0;
		for (int idir = 0; idir < 3; idir++)
		for (int jdir = 0; jdir < 3; jdir++){
			sum_mob_abs_new += fabs(mob_e_new(idir, jdir)) + fabs(mob_h_new(idir, jdir));
			error += fabs(mob_e_new(idir, jdir) - mob_e(idir, jdir)) + fabs(mob_h_new(idir, jdir) - mob_h(idir, jdir));
		}
		rel_error = sum_mob_abs < 1e-20 ? 0 : error / sum_mob_abs;

		iter++;
		if (ionode and(iter <= 5 or(iter < 100 and iter % 10 == 0) or iter % 100 == 0)) printf("iter = %d  error = %8.1le (%11.4le)\n", iter, error, sum_mob_abs);
		cond_e = cond_e_new; cond_h = cond_h_new; cond = cond_new; mob_e = mob_e_new; mob_h = mob_h_new; sum_mob_abs = sum_mob_abs_new;

		if (error < 1e-6 or rel_error < 1e-6) break;
	}
	if (not has_v and sP1 != nullptr){ delete[] P1_; delete[] P2_; P1_ = nullptr; P2_ = nullptr; }

	if (ionode) { printf("\nConverged results (iter %d):\n", iter); fflush(stdout); }
	write_conductivity(cond);
	if (nb > nv) compute_mobility_brange(cond_e, f, nv, nb, "electrons");
	if (nv > 0) compute_mobility_brange(cond_h, f, 0, nv, "holes");

	dealloc_real_array(ApX);
	dealloc_real_array(v);
	dealloc_real_array(imsigp);
	dealloc_real_array(f);
	dealloc_real_array(imsig);
	dealloc_real_array(B);
	dealloc_real_array(dfdEfield);
	if (ionode) { printf("\n"); fflush(stdout); } MPI_Barrier(MPI_COMM_WORLD);
}

matrix3<> electronphonon::compute_conductivity_brange(double **imsigp, double ***v, double **f, int bStart, int bEnd){
	if (ionode) printf("Compute conductivity for bands [%d, %d]:\n", bStart, bEnd);
	matrix3<> cond, mob, Diffusion_Einstein, Diffusion, Diffusion_smaller_dmu;
	for (int ik = 0; ik < nk_glob; ik++)
	for (int b = bStart; b < bEnd; b++){
		double dtmp = imsigp[ik][b] > 1e-40 ? f[ik][b] * (1 - f[ik][b]) / imsigp[ik][b] : 0;
		for (int idir = 0; idir < 3; idir++)
		for (int jdir = 0; jdir < 3; jdir++)
			cond(idir, jdir) += dtmp * v[ik][b][idir] * v[ik][b][jdir];
	}
	cond *= 0.5 / (elec->nk_full * latt->cell_size * elec->temperature);
	return cond;
}

matrix3<> electronphonon::compute_conductivity_brange(double ***dfdEfield, double ***v, int bStart, int bEnd){
	matrix3<> cond;
	for (int ik = 0; ik < nk_glob; ik++)
	for (int b = bStart; b < bEnd; b++){
		for (int idir = 0; idir < 3; idir++)
		for (int jdir = 0; jdir < 3; jdir++)
			cond(idir, jdir) += v[ik][b][idir] * dfdEfield[ik][b][jdir];
	}
	cond *= 1. / (elec->nk_full * latt->cell_size);
	return cond;
}

matrix3<> electronphonon::compute_mobility_brange(matrix3<> cond, double **f, int bStart, int bEnd, string scarr, bool print){
	if (ionode and print) printf("Compute mobility of %s:\n", scarr.c_str());
	matrix3<> mob, Diffusion_Einstein, Diffusion, Diffusion_smaller_dmu;
	double dmu = elec->temperature / 10., dmu_smaller = elec->temperature / 20.;
	if (scarr == "electrons"){
		if (elec->ne < 1e-20) return matrix3<>();
		mob = cond * (1. / elec->ne);
		double ne_dmu = elec->compute_nfree(false, elec->temperature, elec->mu - dmu) / elec->nk_full / latt->cell_size,
			ne_smaller_dmu = elec->compute_nfree(false, elec->temperature, elec->mu - dmu_smaller) / elec->nk_full / latt->cell_size;
		double dnedmu = fabs(elec->ne - ne_dmu) / dmu, dne_smaller_dmu = fabs(elec->ne - ne_smaller_dmu) / dmu_smaller;
		Diffusion = (elec->ne / dnedmu) * mob; Diffusion_smaller_dmu = (elec->ne / dne_smaller_dmu) * mob;
		Diffusion_Einstein = elec->temperature * mob;
	}
	else if (scarr == "holes"){
		if (fabs(elec->nh) < 1e-20) return matrix3<>();
		mob = cond * (1. / fabs(elec->nh));
		double nh_dmu = elec->compute_nfree(true, elec->temperature, elec->mu + dmu) / elec->nk_full / latt->cell_size,
			nh_smaller_dmu = elec->compute_nfree(true, elec->temperature, elec->mu + dmu_smaller) / elec->nk_full / latt->cell_size;
		double dnhdmu = fabs(elec->nh - nh_dmu) / dmu, dnh_smaller_dmu = fabs(elec->nh - nh_smaller_dmu) / dmu_smaller;
		Diffusion = (elec->nh / dnhdmu) * mob; Diffusion_smaller_dmu = (elec->nh / dnh_smaller_dmu) * mob;
		Diffusion_Einstein = elec->temperature * mob;
	}
	else error_message("scarr is not right", "compute_mobility_brange");
	if (ionode and print){
		matrix3<> mob_SI = (1 / cm2byVs2au) * mob;
		mob_SI.print(stdout, "Mobility tensor (cm2/V/s):", latt->dim, "  %10.3le", false, true);
		Diffusion = (1 / cm2bys2au) * Diffusion;
		Diffusion.print(stdout, "Diffusion tensor (cm2/s):", latt->dim, "  %10.3le", false, true);
		Diffusion_smaller_dmu = (1 / cm2bys2au) * Diffusion_smaller_dmu;
		Diffusion_smaller_dmu.print(stdout, "Diffusion tensor (smaller dmu, cm2/s):", latt->dim, "  %10.3le", false, true);
		Diffusion_Einstein = (1 / cm2bys2au) * Diffusion_Einstein;
		Diffusion_Einstein.print(stdout, "Diffusion tensor (Einstein, cm2/s):", latt->dim, "  %10.3le", false, true);
	}
	return mob;
}

void electronphonon::write_conductivity(matrix3<> cond){
	if (ionode && latt->dim >= 2){
		if (latt->dim == 3){
			cond = (1e-9*Ohm*meter) * cond;
			cond.print(stdout, "Conductivity tensor ((nOhm m)^-1):", latt->dim, "  %10.3le", false, true);
			inv(cond).print(stdout, "Resistivity tnesor (nOhm m):", latt->dim, "  %10.3le", false, true);
		}
		if (latt->dim == 2){
			cond = Ohm * cond;
			cond.print(stdout, "Conductivity tensor (Ohm^-1):", latt->dim, "  %10.3le", false, true);
			cond.set(0, 2, 0); cond.set(1, 2, 0); cond.set(2, 0, 0); cond.set(2, 1, 0);
			inv(cond).print(stdout, "Resistivity tnesor (Ohm):", latt->dim, "  %10.3le", false, true);
		}
	}
}