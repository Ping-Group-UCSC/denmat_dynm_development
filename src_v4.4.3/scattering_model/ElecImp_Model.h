#pragma once
#include "common_headers.h"
#include "ScattModel_Param.h"
#include "lattice.h"
#include "parameters.h"
#include "electron.h"
#include "Coulomb_Model.h"
#include "mymp.h"

struct elecimp_model
{
	lattice *latt;
	electron *elec;
	coulomb_model *coul_model;
	int nk, bStart, bEnd, nb, nbpow4, bStart_wannier; // bStart and bEnd relative to bStart_dm
	double nk_full, degauss, ethr, prefac_A, prefac_gauss, prefac_sqrtgauss, prefac_exp_ld, prefac_exp_cv, prefac_imsig;
	double **imsig;
	complex *Uih, *ovlp, *eimp, *P1imp, *P2imp;
	double **e;

	elecimp_model(lattice *latt, parameters *param, electron *elec, int bStart, int bEnd, coulomb_model *coul_model)
		: latt(latt), elec(elec), nk(elec->nk), nk_full(elec->nk_full), 
		bStart(bStart), bEnd(bEnd), nb(bEnd - bStart), nbpow4((int)std::pow(nb, 4)), bStart_wannier(bStart + elec->bStart_dm + elec->bskipped_wannier),
		degauss(param->degauss), ethr(param->degauss*param->ndegauss),
		coul_model(coul_model)
	{
		if (ionode) printf("\nInitialize electron-impurity scattering: %s\n", eip.impMode.c_str());
		if (ionode) printf("bStart = %d bEnd = %d bStart_wannier = %d\n", bStart, bEnd, bStart_wannier);
		if (ionode) printf("ionized impurity denstiy = %10.3le\n", eip.ni_ionized);
		if (eip.ni_ionized <= 0) error_message("eip.ni_ionized must be postive");
		prefac_A = eip.Z * sqrt(eip.ni_ionized * latt->volume);
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

	void calc_A(int ik, int jk, complex* A1, complex* A2){
		double prefac_delta = alg.scatt == "lindblad" ? prefac_sqrtgauss : prefac_gauss;
		double prefac_exp = alg.scatt == "lindblad" ? prefac_exp_ld : prefac_exp_cv;
		calc_eimp(ik, jk);

		// A = Z * sqrt(n*V) e^2 / V / (eps_r * eps_0) / (beta_s^2 + q^2) * <k|k'> * sqrt(delta(ek - ek'))
		// Note that probably A due to different scattering mechanisms can not be sumed directly
		// ImSigma_kn = pi/hbar/Nk * sum_k'n' |A_knk'n'|^2
		for (int b1 = 0; b1 < nb; b1++)
		for (int b2 = 0; b2 < nb; b2++){
			A1[b1*nb + b2] = c0, A2[b1*nb + b2] = c0;
			double de = e[ik][b1] - e[jk][b2];
			if (fabs(de) < ethr){
				double delta = prefac_delta * exp(prefac_exp * std::pow(de, 2)); // prefac_gauss is merged in prefac
				A2[b1*nb + b2] = eimp[b1*nb + b2] * delta;
			}
			A1[b1*nb + b2] = alg.scatt == "lindblad" ? A2[b1*nb + b2] : eimp[b1*nb + b2];
			//double dtmp = prefac_imsig * real(A1[b1*nb + b2] * conj(A2[b1*nb + b2]));
			//if (ik == jk && b1 == b2) dtmp = 0; // a transition between the same state should not contribute to ImSigma
			//imsig[ik][b1] += dtmp; if (ik < jk) imsig[jk][b2] += dtmp;
		}
	}

	void calc_P(int ik, int jk, complex* P1, complex* P2, bool accum = false){
		complex *A1 = new complex[nb*nb]{c0}; complex *A2 = new complex[nb*nb]{c0};
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
					P1imp[n12 + i3*nb + i4] = A1[i13] * conj(A2[i2*nb + i4]);
					P2imp[n12 + i3*nb + i4] = A1[i31] * conj(A2[i4*nb + i2]);
				}
			}
		}
		complex bfac = accum ? c1 : c0;
		axbyc(P1, P1imp, nbpow4, c1, bfac); // y = ax + by + c with a = 1/nk_full and b = 1 and c = 0
		axbyc(P2, P2imp, nbpow4, c1, bfac); // y = ax + by + c with a = 1/nk_full and b = 1 and c = 0

		if (imsig != nullptr) calc_imsig(ik, jk, P1imp, P2imp);
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
			double emax = maxval(e, nk, 0, nb);
			double emin = minval(e, nk, 0, nb);

			string fnamesigkn = "ldbd_imsigkn_ei_" + eip.impMode + "_byDMD.out";
			if (exists(fnamesigkn)) fnamesigkn = "ldbd_imsigkn_ei_" + eip.impMode + "_byDMD_updated.out";
			FILE *fpsigkn = fopen(fnamesigkn.c_str(), "w");
			fprintf(fpsigkn, "E(eV) ImSigma(eV)\n");
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb; b++){
				fprintf(fpsigkn, "%14.7le %14.7le\n", (e[ik][b] - emin) / eV, imsig[ik][b] / eV); fflush(fpsigkn);
			}
			fclose(fpsigkn);

			int ne = 200;
			std::vector<double> imsige(ne+2); std::vector<int> nstate(ne+2);
			double de = (emax - emin) / ne;
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb; b++){
				int ie = round((e[ik][b] - emin) / de);
				if (ie >= 0 && ie <= ne+1){
					nstate[ie]++;
					imsige[ie] += imsig[ik][b];
				}
			}
			string fnamesige = "ldbd_imsige_ei_"+eip.impMode+"_byDMD.out";
			if (exists(fnamesige)) fnamesige = "ldbd_imsige_ei_" + eip.impMode + "_byDMD_updated.out";
			FILE *fpsige = fopen(fnamesige.c_str(), "w");
			fprintf(fpsige, "E(eV) ImSigma(eV) N_States\n");
			for (int ie = 0; ie < ne+2; ie++){
				if (nstate[ie] > 0){
					imsige[ie] /= nstate[ie];
					fprintf(fpsige, "%14.7le %14.7le %d\n", ie*de / eV, imsige[ie] / eV, nstate[ie]); fflush(fpsige);
				}
			}
			fclose(fpsige);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		dealloc_real_array(imsig);
	}
};