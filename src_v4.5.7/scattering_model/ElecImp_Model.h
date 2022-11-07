#pragma once
#include "common_headers.h"
#include "Scatt_Param.h"
#include "lattice.h"
#include "parameters.h"
#include "electron.h"
#include "Coulomb_Model.h"
#include "mymp.h"

struct elecimp_model
{
	int iD;
	lattice *latt;
	electron *elec;
	coulomb_model *coul_model;
	int nk, bStart, bEnd, nb, nbpow4, bStart_wannier; // bStart and bEnd relative to bStart_dm
	double nk_full, degauss, ethr, prefac_A, prefac_gauss, prefac_sqrtgauss, prefac_exp_ld, prefac_exp_cv, prefac_imsig;
	double **imsig;
	complex *Uih, *ovlp, *eimp, *P1imp, *P2imp, *A1, *A2, *A1pp, *A1pm, *A1mp, *A1mm, *A2pp, *A2pm, *A2mp, *A2mm;
	double **e, eStart, eEnd, omegaL;

	elecimp_model(int iD, lattice *latt, parameters *param, electron *elec, int bStart, int bEnd, double eStart, double eEnd, coulomb_model *coul_model)
		: iD(iD), latt(latt), elec(elec), nk(elec->nk), nk_full(elec->nk_full), 
		bStart(bStart), bEnd(bEnd), nb(bEnd - bStart), nbpow4((int)std::pow(nb, 4)), bStart_wannier(bStart + elec->bStart_dm + elec->bskipped_wannier),
		eStart(eStart), eEnd(eEnd),
		degauss(eip.degauss[iD]), ethr(param->degauss*param->ndegauss), omegaL(1e-4*elec->temperature),
		coul_model(coul_model)
	{
		if (ionode) printf("\nInitialize electron-impurity scattering: %s\n", eip.impMode[iD].c_str());
		if (ionode) printf("bStart = %d bEnd = %d bStart_wannier = %d\n", bStart, bEnd, bStart_wannier);
		if (ionode) printf("ionized impurity denstiy = %10.3le\n", eip.ni_ionized[iD]);
		if (eip.ni_ionized[iD] <= 0) error_message("eip.ni_ionized[iD] must be postive");
		prefac_A = eip.Z[iD] * sqrt(eip.ni_ionized[iD] * latt->cell_size);
		prefac_gauss = 1. / (degauss * sqrt(2.*M_PI));
		prefac_sqrtgauss = sqrt(prefac_gauss);
		prefac_exp_cv = -0.5 / std::pow(degauss, 2);
		prefac_exp_ld = -0.25 / std::pow(degauss, 2);
		prefac_imsig = M_PI / nk_full;

		imsig = alloc_real_array(nk, nb);
		Uih = new complex[nb*elec->nb_wannier]{c0};
		ovlp = new complex[nb*nb];
		eimp = new complex[nb*nb];
		P1imp = new complex[nbpow4]{c0}; P2imp = new complex[nbpow4]{c0};
		if (!eip.detailBalance[iD]){
			A1 = new complex[nb*nb]{c0}; A2 = new complex[nb*nb]{c0};
		}
		else{
			A1pp = new complex[nb*nb]{c0}; A2pp = new complex[nb*nb]{c0}; A1pm = new complex[nb*nb]{c0}; A2pm = new complex[nb*nb]{c0};
			A1mp = new complex[nb*nb]{c0}; A2mp = new complex[nb*nb]{c0}; A1mm = new complex[nb*nb]{c0}; A2mm = new complex[nb*nb]{c0};
		}

		if (bStart != coul_model->bStart || bEnd != coul_model->bEnd) error_message("bStart(bEnd) must be the same as bStart in coul_model","elecimp_model");
		e = coul_model->e;
	}
	
	void calc_ovlp(int ik, int jk){
		hermite(elec->U[ik], Uih, elec->nb_wannier, nb);
		zgemm_interface(ovlp, Uih, elec->U[jk], nb, nb, elec->nb_wannier);
	}
	void calc_eimp(int ik, int jk){
		calc_ovlp(ik, jk);
		axbyc(eimp, ovlp, nb*nb, prefac_A * coul_model->vq(elec->kvec[ik] - elec->kvec[jk])); // y = ax
	}
	void calc_eimp_debug(int ik, int jk){
		ostringstream convert; convert << mpkpair.myrank;
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
		FILE *fp; string fname = dir_debug + "debug_calc_eimp.out." + convert.str();
		bool ldebug = DEBUG;
		if (ldebug) fp = fopen(fname.c_str(), "a");
		MPI_Barrier(MPI_COMM_WORLD);

		calc_ovlp(ik, jk);
		if (ldebug){
			fprintf_complex_mat(fp, ovlp, nb, "ovlp:"); fflush(fp);
		}
		axbyc(eimp, ovlp, nb*nb, prefac_A * coul_model->vq(elec->kvec[ik] - elec->kvec[jk])); // y = ax
		if (ldebug){
			fprintf(fp, "|q|= %lg vq = %lg prefac_A = %lg\n", 
				latt->GGT.metric_length_squared(wrap(elec->kvec[ik] - elec->kvec[jk])), 
				real(coul_model->vq(elec->kvec[ik] - elec->kvec[jk])), prefac_A);
			fprintf_complex_mat(fp, eimp, nb, "eimp:"); fflush(fp);
		}
		if (ldebug) fclose(fp);
	}

	void calc_A(int ik, int jk, complex* A1, complex* A2){
		double prefac_delta = alg.scatt == "lindblad" ? prefac_sqrtgauss : prefac_gauss;
		if (eip.detailBalance[iD]) prefac_delta *= 0.5;
		double prefac_exp = alg.scatt == "lindblad" ? prefac_exp_ld : prefac_exp_cv;
		calc_eimp(ik, jk);

		// A = Z * sqrt(n*V) e^2 / V / (eps_r * eps_0) / (beta_s^2 + q^2) * <k|k'> * sqrt(delta(ek - ek'))
		// Note that probably A due to different scattering mechanisms can not be sumed directly
		// ImSigma_kn = pi/hbar/Nk * sum_k'n' |A_knk'n'|^2
		for (int b1 = 0; b1 < nb; b1++)
		for (int b2 = 0; b2 < nb; b2++){
			if (!eip.detailBalance[iD]){
				A1[b1*nb + b2] = c0, A2[b1*nb + b2] = c0;
				double de = e[ik][b1] - e[jk][b2];
				if (fabs(de) < ethr){
					double delta = prefac_delta * exp(prefac_exp * std::pow(de, 2)); // prefac_gauss is merged in prefac
					A2[b1*nb + b2] = eimp[b1*nb + b2] * delta;
				}
				A1[b1*nb + b2] = alg.scatt == "lindblad" ? A2[b1*nb + b2] : eimp[b1*nb + b2];
			}
			else{
				A1pp[b1*nb + b2] = c0, A2pp[b1*nb + b2] = c0; A1pm[b1*nb + b2] = c0, A2pm[b1*nb + b2] = c0;
				A1mp[b1*nb + b2] = c0, A2mp[b1*nb + b2] = c0; A1mm[b1*nb + b2] = c0, A2mm[b1*nb + b2] = c0;
				double de = e[ik][b1] - e[jk][b2];
				complex G1p = c0, G2p = c0, G1m = c0, G2m = c0;
				// emission
				if (fabs(de + omegaL) < ethr){
					double deltaplus = prefac_delta * exp(prefac_exp * std::pow(de + omegaL, 2)); // prefac_gauss is merged in prefac
					G2p = eimp[b1*nb + b2] * deltaplus;
				}
				G1p = alg.scatt == "lindblad" ? G2p : eimp[b1*nb + b2];
				double dEbyT = de / elec->temperature;
				double facDB = (-dEbyT < 46) ? exp(-dEbyT / 2) : 1; //when Ek + wqp = Ekp, nq + 1 = exp[(Ekp - Ek)/T] * nq
				double facDB2 = 1;
				A1pp[b1*nb + b2] = G1p * facDB;
				A1pm[b1*nb + b2] = G1p * facDB2;
				A2pp[b1*nb + b2] = G2p * facDB;
				A2pm[b1*nb + b2] = G2p * facDB2;
				// absorption
				if (fabs(de - omegaL) < ethr){
					double deltaminus = prefac_delta * exp(prefac_exp * std::pow(de - omegaL, 2)); // prefac_gauss is merged in prefac
					G2m = eimp[b1*nb + b2] * deltaminus;
				}
				G1m = alg.scatt == "lindblad" ? G2m : eimp[b1*nb + b2];
				facDB = (dEbyT < 46) ? exp(dEbyT / 2) : 1; //when Ek + wqp = Ekp, nq + 1 = exp[(Ekp - Ek)/T] * nq
				facDB2 = 1;
				A1mp[b1*nb + b2] = G1m * facDB;
				A1mm[b1*nb + b2] = G1m * facDB2;
				A2mp[b1*nb + b2] = G2m * facDB;
				A2mm[b1*nb + b2] = G2m * facDB2;
			}
		}
	}
	void calc_A_debug(int ik, int jk, complex* A1, complex* A2){
		ostringstream convert; convert << mpkpair.myrank;
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
		FILE *fp; string fname = dir_debug + "debug_calc_Aimp.out." + convert.str();
		bool ldebug = DEBUG;
		if (ldebug) fp = fopen(fname.c_str(), "a");
		MPI_Barrier(MPI_COMM_WORLD);

		double prefac_delta = alg.scatt == "lindblad" ? prefac_sqrtgauss : prefac_gauss;
		double prefac_exp = alg.scatt == "lindblad" ? prefac_exp_ld : prefac_exp_cv;
		calc_eimp_debug(ik, jk);

		if (ldebug){
			fprintf(fp, "prefac_delta = %lg, prefac_exp = %lg\n", prefac_delta, prefac_exp);
			fprintf_complex_mat(fp, eimp, nb, "eimp:"); fflush(fp);
		}

		// A = Z * sqrt(n*V) e^2 / V / (eps_r * eps_0) / (beta_s^2 + q^2) * <k|k'> * sqrt(delta(ek - ek'))
		// Note that probably A due to different scattering mechanisms can not be sumed directly
		// ImSigma_kn = pi/hbar/Nk * sum_k'n' |A_knk'n'|^2
		for (int b1 = 0; b1 < nb; b1++)
		for (int b2 = 0; b2 < nb; b2++){
			if (!eip.detailBalance[iD]){
				A1[b1*nb + b2] = c0, A2[b1*nb + b2] = c0;
				double de = e[ik][b1] - e[jk][b2];
				if (fabs(de) < ethr){
					double delta = prefac_delta * exp(prefac_exp * std::pow(de, 2)); // prefac_gauss is merged in prefac
					A2[b1*nb + b2] = eimp[b1*nb + b2] * delta;
				}
				if (ldebug){
					double delta = prefac_delta * exp(prefac_exp * std::pow(de, 2));
					fprintf(fp, "%lg %lg\n", de, delta);
				}
				A1[b1*nb + b2] = alg.scatt == "lindblad" ? A2[b1*nb + b2] : eimp[b1*nb + b2];
			}
		}
		if (ldebug){
			if (!eip.detailBalance[iD]){
				fprintf_complex_mat(fp, A1, nb, "A1:"); fflush(fp);
				fprintf_complex_mat(fp, A2, nb, "A2:"); fflush(fp);
			}
		}
		if (ldebug) fclose(fp);
	}

	void calc_P(int ik, int jk, complex* P1, complex* P2, bool accum = false){
		calc_A(ik, jk, A1, A2);

		// P1_n3n2,n4n5 = A_n3n4 * conj(A_n2n5)
		// P2_n3n4,n1n5 = A_n1n3 * conj(A_n5n4)
		// P due to e-ph and e-i scatterings can be sumed directly, I think
		for (int i1 = 0; i1 < nb; i1++)
		for (int i2 = 0; i2 < nb; i2++){
			int n12 = (i1*nb + i2)*nb*nb;
			for (int i3 = 0; i3 < nb; i3++){
				int i13 = i1*nb + i3;
				int i31 = i3*nb + i1;
				for (int i4 = 0; i4 < nb; i4++){
					if (!eip.detailBalance[iD]){
						P1imp[n12 + i3*nb + i4] = A1[i13] * conj(A2[i2*nb + i4]);
						P2imp[n12 + i3*nb + i4] = A1[i31] * conj(A2[i4*nb + i2]);
					}
					else{
						P1imp[n12 + i3*nb + i4] = A1pp[i13] * conj(A2pp[i2*nb + i4]) + A1mm[i13] * conj(A2mm[i2*nb + i4]);
						P2imp[n12 + i3*nb + i4] = A1mp[i31] * conj(A2mp[i4*nb + i2]) + A1pm[i31] * conj(A2pm[i4*nb + i2]);
					}
				}
			}
		}
		complex bfac = accum ? c1 : c0;
		axbyc(P1, P1imp, nbpow4, c1, bfac); // y = ax + by + c with a = 1 and b = bfac and c = 0
		axbyc(P2, P2imp, nbpow4, c1, bfac); // y = ax + by + c with a = 1 and b = bfac and c = 0

		if (imsig != nullptr) calc_imsig(ik, jk, P1imp, P2imp);
	}
	void calc_P_debug(int ik, int jk, complex* P1, complex* P2, bool accum = false){
		ostringstream convert; convert << mpkpair.myrank;
		convert.flush(); MPI_Barrier(MPI_COMM_WORLD); // seems necessary! Otherwise fname is not created for non-root processes
		FILE *fp; string fname = dir_debug + "debug_calc_Pimp.out." + convert.str();
		bool ldebug = DEBUG;
		if (ldebug) fp = fopen(fname.c_str(), "a");
		MPI_Barrier(MPI_COMM_WORLD);

		calc_A_debug(ik, jk, A1, A2);
		if (ldebug){
			fprintf_complex_mat(fp, A1, nb, "A1:"); fflush(fp);
			fprintf_complex_mat(fp, A2, nb, "A2:"); fflush(fp);
		}

		// P1_n3n2,n4n5 = A_n3n4 * conj(A_n2n5)
		// P2_n3n4,n1n5 = A_n1n3 * conj(A_n5n4)
		// P due to e-ph and e-i scatterings can be sumed directly, I think
		for (int i1 = 0; i1 < nb; i1++)
		for (int i2 = 0; i2 < nb; i2++){
			int n12 = (i1*nb + i2)*nb*nb;
			for (int i3 = 0; i3 < nb; i3++){
				int i13 = i1*nb + i3;
				int i31 = i3*nb + i1;
				for (int i4 = 0; i4 < nb; i4++){
					P1imp[n12 + i3*nb + i4] = A1[i13] * conj(A2[i2*nb + i4]);
					P2imp[n12 + i3*nb + i4] = A1[i31] * conj(A2[i4*nb + i2]);
				}
			}
		}
		complex bfac = accum ? c1 : c0;
		axbyc(P1, P1imp, nbpow4, c1, bfac); // y = ax + by + c with a = 1 and b = bfac and c = 0
		axbyc(P2, P2imp, nbpow4, c1, bfac); // y = ax + by + c with a = 1 and b = bfac and c = 0

		if (imsig != nullptr) calc_imsig(ik, jk, P1imp, P2imp);
		if (ldebug){
			fprintf_complex_mat(fp, P1imp, nb*nb, "P1:"); fflush(fp);
			fprintf_complex_mat(fp, P2imp, nb*nb, "P2:"); fflush(fp);
			fprintf_real_array(fp, imsig[ik], nb, "imsig]ik]:"); fflush(fp);
			fprintf_real_array(fp, imsig[jk], nb, "imsig]jk]:"); fflush(fp);
		}
		if (ldebug) fclose(fp);
	}
	void calc_imsig(int ik, int jk, complex* P1, complex* P2){
		for (int b1 = 0; b1 < nb; b1++){
			int n11 = (b1*nb + b1)*nb*nb;
			for (int b2 = 0; b2 < nb; b2++){
				double dtmp = prefac_imsig * real(P1[n11 + b2*nb + b2]);
				if (ik == jk && b1 == b2) dtmp = 0; // a transition between the same state will not contribute to ImSigma
				imsig[ik][b1] += dtmp; if (ik < jk) imsig[jk][b2] += dtmp;
			}
		}
	}

	void reduce_imsig(mymp *mp){
		if (imsig == nullptr) return;
		mp->allreduce(imsig, nk, nb, MPI_SUM);

		if (ionode){
			string fnamesigkn = "ldbd_imsigkn_ei_D" + int2str(iD) + eip.impMode[iD] + "_byDMD.out";
			if (exists(fnamesigkn)) fnamesigkn = "ldbd_imsigkn_ei_" + int2str(iD) + eip.impMode[iD] + "_byDMD_updated.out";
			FILE *fpsigkn = fopen(fnamesigkn.c_str(), "w");
			fprintf(fpsigkn, "E(eV) ImSigma(eV)\n");
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb; b++){
				fprintf(fpsigkn, "%14.7le %14.7le\n", e[ik][b] / eV, imsig[ik][b] / eV); fflush(fpsigkn);
			}
			fclose(fpsigkn);

			int ne = 200;
			std::vector<double> imsige(ne+2); std::vector<int> nstate(ne+2);
			double de = (eEnd - eStart) / ne;
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb; b++){
				int ie = round((e[ik][b] - eStart) / de);
				if (ie >= 0 && ie <= ne+1){
					nstate[ie]++;
					imsige[ie] += imsig[ik][b];
				}
			}
			string fnamesige = "ldbd_imsige_ei_D" + int2str(iD+1) + "_" + eip.impMode[iD] + "_byDMD.out";
			if (exists(fnamesige)) fnamesige = "ldbd_imsige_ei_D" + int2str(iD+1) + "_" + eip.impMode[iD] + "_byDMD_updated.out";
			FILE *fpsige = fopen(fnamesige.c_str(), "w");
			fprintf(fpsige, "E(eV) ImSigma(eV) N_States\n");
			for (int ie = 0; ie < ne+2; ie++){
				if (nstate[ie] > 0){
					imsige[ie] /= nstate[ie];
					fprintf(fpsige, "%14.7le %14.7le %d\n", (eStart + ie*de) / eV, imsige[ie] / eV, nstate[ie]); fflush(fpsige);
				}
			}
			fclose(fpsige);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		dealloc_real_array(imsig);
	}
};