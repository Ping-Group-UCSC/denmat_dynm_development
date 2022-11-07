#pragma once
#include "common_headers.h"
#include "Scatt_Param.h"
#include "lattice.h"
#include "parameters.h"
#include "electron.h"
#include "Coulomb_Model.h"
#include "mymp.h"

struct elecelec_model
{
	lattice *latt;
	electron *elec;
	coulomb_model *coul_model;
	int nk, bStart, bEnd, nb, nbpow4, bStart_wannier; // bStart and bEnd relative to bStart_dm
	double nk_full, degauss, ethr, prefac_gauss, prefac_sqrtgauss, prefac_exp_ld, prefac_exp_cv, prefac_imsig;
	double **imsig, *delta;
	complex *Uih, *ovlp12, *ovlp34, *ovlp32, *ovlp14, *mee, *mee_ex, *A1, *A2, *A1rho, *A1rhobar, *P1ee, *P2ee, *mtmp, *ddml, *ddmr, *A1rho_dP, *A1rhobar_dP;
	double **e, **f, eStart, eEnd;
	kIndexMap *kmap;

	elecelec_model(lattice *latt, parameters *param, electron *elec, int bStart, int bEnd, double eStart, double eEnd, coulomb_model *coul_model)
		: latt(latt), elec(elec), nk(elec->nk), nk_full(elec->nk_full), 
		bStart(bStart), bEnd(bEnd), nb(bEnd - bStart), nbpow4((int)std::pow(nb, 4)), bStart_wannier(bStart + elec->bStart_dm + elec->bskipped_wannier),
		eStart(eStart), eEnd(eEnd),
		degauss(eep.degauss), ethr(param->degauss*param->ndegauss),
		coul_model(coul_model), kmap(nullptr)
	{
		if (ionode) printf("\nInitialize electron-electron scattering: %s\n", eep.eeMode.c_str());
		if (ionode) printf("bStart = %d bEnd = %d bStart_wannier = %d\n", bStart, bEnd, bStart_wannier);
		prefac_gauss = 1. / (degauss * sqrt(2.*M_PI));
		prefac_sqrtgauss = sqrt(prefac_gauss);
		prefac_exp_cv = -0.5 / std::pow(degauss, 2);
		prefac_exp_ld = -0.25 / std::pow(degauss, 2);
		prefac_imsig = M_PI / nk_full;

		imsig = alloc_real_array(nk, nb);
		Uih = new complex[nb*elec->nb_wannier]{c0};
		ovlp12 = new complex[nb*nb]{c0}; ovlp34 = new complex[nb*nb]{c0};
		ovlp32 = new complex[nb*nb]{c0}; ovlp14 = new complex[nb*nb]{c0};
		delta = new double[nbpow4]{0};
		mee = new complex[nbpow4]{c0};
		if (eep.antisymmetry) mee_ex = new complex[nbpow4]{c0};
		A1 = new complex[nbpow4]{c0}; A2 = new complex[nbpow4]{c0};
		A1rho = new complex[nb*nb]{c0}; A1rhobar = new complex[nb*nb]{c0};
		mtmp = new complex[nb*nb]{c0};
		P1ee = new complex[nbpow4]{c0}; P2ee = new complex[nbpow4]{c0};
		if (alg.linearize_dPee){
			ddml = new complex[nb*nb]{c0}; ddmr = new complex[nb*nb]{c0};
			A1rho_dP = new complex[nb*nb]{c0}; A1rhobar_dP = new complex[nb*nb]{c0};
		}

		if (bStart != coul_model->bStart || bEnd != coul_model->bEnd) error_message("bStart(bEnd) must be the same as bStart in coul_model", "elecelec_model");
		this->e = coul_model->e; this->f = coul_model->f;
		
		kmap = new kIndexMap(elec->kmesh, elec->kvec);
		if (ionode && DEBUG){
			string fnamek = dir_debug + "kIndexMap.out";
			FILE *fpk = fopen(fnamek.c_str(), "w");
			fprintf(fpk, "\nPrint kIndexMap:\n");
			for (size_t ik = 0; ik < elec->kvec.size(); ik++){
				std::map<vector3<int>, size_t>::iterator iter = kmap->the_map.find(kmap->ikvec3(elec->kvec[ik]));
				fprintf(fpk, "ikvec3 = (%d,%d,%d) ik = %lu\n", iter->first[0], iter->first[1], iter->first[2], iter->second);
			}
			fclose(fpk);
		}
	}
	
	void calc_ovlps(int ik1, int ik2, int ik3, int ik4, bool calc_ovlp12){
		if (calc_ovlp12) calc_ovlp(ovlp12, ik1, ik2, true);
		calc_ovlp(ovlp34, ik3, ik4);
		if (!eep.antisymmetry) return;
		calc_ovlp(ovlp14, ik1, ik4, true);
		calc_ovlp(ovlp32, ik3, ik2);
	}
	void calc_ovlp(complex *ovlp, int ik, int jk, bool multply_vq = false){
		hermite(elec->U[ik], Uih, elec->nb_wannier, nb);
		zgemm_interface(ovlp, Uih, elec->U[jk], nb, nb, elec->nb_wannier);

		if (!multply_vq) return;
		if (clp.dynamic == "static"){
			complex vq = coul_model->vq(elec->kvec[ik] - elec->kvec[jk]);
			axbyc(ovlp, nullptr, nb*nb, c0, vq);
		}
	}
	void calc_mee(int ik1 = 0, int ik2 = 0, int ik3 = 0, int ik4 = 0){
		vector3<> q(elec->kvec[ik1] - elec->kvec[ik2]);
		// mee = <1|<3| vq |2>|4> = vq <1|2> <3|4>
		for (int i1 = 0; i1 < nb; i1++)
		for (int i2 = 0; i2 < nb; i2++){
			int i12 = i1*nb + i2;
			int n12 = i12*nb*nb;
			for (int i3 = 0; i3 < nb; i3++)
			for (int i4 = 0; i4 < nb; i4++){
				mee[n12 + i3*nb + i4] = ovlp12[i12] * ovlp34[i3*nb + i4];
				if (clp.dynamic != "static"){
					complex vq;
					if (clp.dynamic_screening_ee_two_freqs){
						double w12 = e[ik1][i1] - e[ik2][i2], w43 = e[ik4][i4] - e[ik3][i3];
						complex vq12 = coul_model->vq(q, w12), vq43 = coul_model->vq(q, w43);
						double vq_abs = sqrt(vq12.abs() * vq43.abs()),
							vq_arg = 0.5*(vq12.arg() + vq43.arg());
						vq = vq_abs * cis(vq_arg);
					}
					else{
						double w = 0.5*(e[ik1][i1] - e[ik2][i2] + e[ik4][i4] - e[ik3][i3]);
						vq = coul_model->vq(q, w);
					}
					mee[n12 + i3*nb + i4]  *= vq;
				}
			}
		}
	}
	void calc_mee_ex(int ik1 = 0, int ik2 = 0, int ik3 = 0, int ik4 = 0){
		vector3<> q(elec->kvec[ik1] - elec->kvec[ik4]);
		// mee_ex = <1|<3| vq |4>|2> = vq <1|4> <3|2>
		for (int i1 = 0; i1 < nb; i1++)
		for (int i2 = 0; i2 < nb; i2++){
			int n12 = (i1*nb + i2)*nb*nb;
			for (int i3 = 0; i3 < nb; i3++){
				int i32 = i3*nb + i2;
				for (int i4 = 0; i4 < nb; i4++){
					mee_ex[n12 + i3*nb + i4] = ovlp14[i1*nb + i4] * ovlp32[i32];
					if (clp.dynamic != "static"){
						complex vq;
						if (clp.dynamic_screening_ee_two_freqs){
							double w14 = e[ik1][i1] - e[ik4][i4], w23 = e[ik2][i2] - e[ik3][i3];
							complex vq14 = coul_model->vq(q, w14), vq23 = coul_model->vq(q, w23);
							double vq_abs = sqrt(vq14.abs() * vq23.abs()),
								vq_arg = 0.5*(vq14.arg() + vq23.arg());
							vq = vq_abs * cis(vq_arg);
						}
						else{
							double w = 0.5*(e[ik1][i1] - e[ik4][i4] + e[ik2][i2] - e[ik3][i3]);
							vq = coul_model->vq(q, w);
						}
						mee[n12 + i3*nb + i4] *= vq;
					}
				}
			}
		}
	}

	bool calc_A(int ik1, int ik2, int ik3, int ik4, bool calc_ovlp12){
		if (!calc_delta(ik1, ik2, ik3, ik4)) return false;
		calc_ovlps(ik1, ik2, ik3, ik4, calc_ovlp12);
		calc_mee(ik1, ik2, ik3, ik4);
		if (eep.antisymmetry){
			calc_mee_ex(ik1, ik2, ik3, ik4);
			axbyc(mee, mee_ex, nbpow4, complex(-0.5,0), complex(0.5,0)); // mee = (mee - mee_ex)/2
		}

		// A = mee delta(e1+e3-e2-e4)
		for (int i1 = 0; i1 < nb; i1++)
		for (int i2 = 0; i2 < nb; i2++){
			int n12 = (i1*nb + i2)*nb*nb;
			for (int i3 = 0; i3 < nb; i3++)
			for (int i4 = 0; i4 < nb; i4++){
				int i1234 = n12 + i3*nb + i4;
				A2[i1234] = delta[i1234] * mee[i1234];
				A1[i1234] = alg.scatt == "lindblad" ? A2[i1234] : mee[i1234];
			}
		}
		return true;
	}
	bool calc_delta(int ik1, int ik2, int ik3, int ik4){
		double prefac_exp = alg.scatt == "lindblad" ? prefac_exp_ld : prefac_exp_cv;
		bool concerved = false;
		for (int i1 = 0; i1 < nb; i1++)
		for (int i2 = 0; i2 < nb; i2++){
			int n12 = (i1*nb + i2)*nb*nb;
			for (int i3 = 0; i3 < nb; i3++)
			for (int i4 = 0; i4 < nb; i4++){
				int i1234 = n12 + i3*nb + i4;
				double de = e[ik1][i1] - e[ik2][i2] + e[ik3][i3] - e[ik4][i4];
				if (fabs(de) < ethr){
					delta[i1234] = exp(prefac_exp * std::pow(de, 2)); // multiply prefac_gauss when computing P?ee
					concerved = true;
				}
				else delta[i1234] = 0;
			}
		}
		return concerved;
	}

	int calc_P(int ik1, int ik2, complex* P1, complex* P2, bool accum = false, complex **dm = nullptr, complex **dm1 = nullptr, 
		complex *dP1 = nullptr, complex *dP2 = nullptr, double **f_eq = nullptr, double **f1_eq = nullptr){
		// P1_{(k1,b1,b2),(k2,b3,b4)} = sum_{k3,b5,b6,b7,b8} [ (I-rho)_{k3,b6,b5} A1(k1-k2)_{(k1,b1),(k2,b3),(k3,b5),(k1+k3-k2,b7) rho_{k1+k3-k2,b7,b8}}
		//                                                   * A2(k1-k2)^*_{(k1,b2),(k2,b4),(k3,b6),(k1+k3-k2,b8) ]
		// P2_{(k1,b1,b2),(k2,b3,b4)} = P1_{(k2,b1,b2),(k1,b3,b4)}^*
		//                            = sum_{k3,b5,b6,b7,b8} [ rho_{k3,b6,b5} A1(k2-k1)_{(k1,b3),(k2,b1),(k3,b5),(k1+k3-k2,b7) (I-rho)_{k1+k3-k2,b7,b8}}
		//                                                   * A2(k2-k1)^*_{(k1,b4),(k2,b2),(k3,b6),(k1+k3-k2,b8) ]
		if (ik1 == ik2 && !eep.antisymmetry){
			if (!accum) { zeros(P1, nbpow4); zeros(P2, nbpow4); }
			return 0;
		}
		zeros(P1ee, nbpow4); zeros(P2ee, nbpow4);
		if (alg.linearize_dPee){ zeros(dP1, nbpow4); zeros(dP2, nbpow4); }
		calc_ovlp(ovlp12, ik1, ik2, true); // <1|2> multiplied by vq

		int nk3_count = 0;
		for (int ik3 = 0; ik3 < nk; ik3++){
			if (eep.antisymmetry && ik1 == ik2 && ik2 == ik3) continue;
			size_t ik4 = 0; vector3<> k4 = elec->kvec[ik1] + elec->kvec[ik3] - elec->kvec[ik2]; // not necessage to wrap k point around Gamma, kmap subroutines will wrap inside
			if (kmap->findk(k4, ik4)){
				// note that for A, I exchanged the 2nd and 3rd index compared with standard definition
				// here A_1234 = <1|vq|2> <3|4> delta(e1+e3-e2-e4)^0.5
				// while in paper, A_1234 = sqrt(2pi) <1|vq|3> <2|4> delta(e1+e2-e3-e4)^0.5
				if (!calc_A(ik1, ik2, ik3, (int)ik4, false)) continue; // if energy conservation is not satisfied
				nk3_count++;
				for (int i1 = 0; i1 < nb; i1++)
				for (int i3 = 0; i3 < nb; i3++){
					if (alg.linearize_dPee){
						calc_A1f(A1rho, (complex *)&A1[(i1*nb + i3)*nb*nb], f_eq[ik3], f_eq[ik4]);
						calc_A1fbar(A1rhobar, (complex *)&A1[(i3*nb + i1)*nb*nb], f_eq[ik3], f_eq[ik4]);
						calc_A1rho_dP(A1rho_dP, (complex *)&A1[(i1*nb + i3)*nb*nb], dm1[ik3], f1_eq[ik3], dm[ik4], f_eq[ik4]);
						calc_A1rho_dP(A1rhobar_dP, (complex *)&A1[(i3*nb + i1)*nb*nb], dm[ik3], f_eq[ik3], dm1[ik4], f1_eq[ik4]);
					}
					else{
						if (dm == nullptr){
							calc_A1f(A1rho, (complex *)&A1[(i1*nb + i3)*nb*nb], f[ik3], f[ik4]);
							calc_A1fbar(A1rhobar, (complex *)&A1[(i3*nb + i1)*nb*nb], f[ik3], f[ik4]);
						}
						else{
							calc_A1rho(A1rho, (complex *)&A1[(i1*nb + i3)*nb*nb], dm1[ik3], dm[ik4]);
							calc_A1rho(A1rhobar, (complex *)&A1[(i3*nb + i1)*nb*nb], dm[ik3], dm1[ik4]);
						}
					}

					for (int i2 = 0; i2 < nb; i2++){
						int n12 = (i1*nb + i2)*nb*nb;
						for (int i4 = 0; i4 < nb; i4++){
							int i1234 = n12 + i3*nb + i4;
							int n24 = (i2*nb + i4)*nb*nb;
							int n42 = (i4*nb + i2)*nb*nb;
							for (int i6 = 0; i6 < nb; i6++)
							for (int i8 = 0; i8 < nb; i8++){
								int i68 = i6*nb + i8;
								P1ee[i1234] += A1rho[i68] * conj(A2[n24 + i68]);
								P2ee[i1234] += A1rhobar[i68] * conj(A2[n42 + i68]);
								if (alg.linearize_dPee){
									dP1[i1234] += A1rho_dP[i68] * conj(A2[n24 + i68]);
									dP2[i1234] += A1rhobar_dP[i68] * conj(A2[n42 + i68]);
								}
							}
						}
					}
				}
			}
		}
		complex prefac = complex(prefac_gauss / nk_full, 0);
		if (eep.antisymmetry) prefac *= 2;
		axbyc(P1ee, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0
		axbyc(P2ee, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0
		if (alg.linearize_dPee) axbyc(dP1, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0
		if (alg.linearize_dPee) axbyc(dP2, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0
		complex bfac = accum ? c1 : c0;
		axbyc(P1, P1ee, nbpow4, c1, bfac); // y = ax + by + c with a = 1/nk_full and b = 1 and c = 0
		axbyc(P2, P2ee, nbpow4, c1, bfac); // y = ax + by + c with a = 1/nk_full and b = 1 and c = 0

		if (imsig != nullptr) calc_imsig(ik1, ik2, P1ee, P2ee);
		return nk3_count;
	}
	void calc_A1rho(complex *A1rho, complex *A1, complex *dmleft, complex *dmright){
		// A1rho = dmleft * A1 * dmright
		zhemm_interface(mtmp, false, dmright, A1, nb);
		zhemm_interface(A1rho, true, dmleft, mtmp, nb);
	}
	void calc_A1rho_dP(complex *A1rho_dP, complex *A1, complex *dmleft, double *fl, complex *dmright, double *fr){
		// A1rho = (dmleft - fl) * A1 * fr + fl * A1 * (dmright - fr)
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++){
			ddml[i*nb + j] = dmleft[i*nb + j] - fl[i];
			ddmr[i*nb + j] = dmright[i*nb + j] - fr[i];
		}
		zhemm_interface(mtmp, false, ddmr, A1, nb);
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			A1rho_dP[i*nb + j] = fl[i] * mtmp[i*nb + j];
		zhemm_interface(mtmp, true, ddml, A1, nb);
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			A1rho_dP[i*nb + j] += mtmp[i*nb + j] * fr[j];
	}
	void calc_A1f(complex *A1f, complex *A1, double* fl, double* fr){
		// A1f = (1-fl) * A1 * fr
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			A1f[i*nb + j] = (1 - fl[i]) * A1[i*nb + j] * fr[j];
	}
	void calc_A1fbar(complex *A1fbar, complex *A1, double* fl, double* fr){
		// A1f = fl * A1 * (1-fr)
		for (int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++)
			A1fbar[i*nb + j] = fl[i] * A1[i*nb + j] * (1 - fr[j]);
	}

	int calc_P3P4(int ik1, int ika, complex* P3, complex* P4);
	int calc_P5P6(int ik1, int ika, complex* P5, complex* P6);

	void calc_imsig(int ik, int jk, complex* P1, complex* P2){
		for (int b1 = 0; b1 < nb; b1++){
			int i11 = b1*nb + b1;
			int n11 = i11*nb*nb;
			for (int b2 = 0; b2 < nb; b2++){
				if (ik == jk && b1 == b2) continue;
				int i22 = b2*nb + b2;
				int n22 = i22*nb*nb;
				imsig[ik][b1] += prefac_imsig * (real(P1[n11 + i22]) * f[jk][b2] + real(P2[n22 + i11]) * (1 - f[jk][b2]));
				if (ik < jk) imsig[jk][b2] += prefac_imsig * (real(P2[n22 + i11]) * f[ik][b1] + real(P1[n11 + i22]) * (1 - f[ik][b1]));
			}
		}
	}

	void reduce_imsig(mymp *mp){
		if (imsig == nullptr) return;
		mp->allreduce(imsig, nk, nb, MPI_SUM);

		if (ionode){
			string fnamesigkn = "ldbd_imsigkn_ee_" + eep.eeMode + "_byDMD.out";
			if (exists(fnamesigkn)) fnamesigkn = "ldbd_imsigkn_ee_" + eep.eeMode + "_byDMD_updated.out";
			FILE *fpsigkn = fopen(fnamesigkn.c_str(), "w");
			fprintf(fpsigkn, "E(eV) ImSigma(eV)\n");
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nb; b++){
				fprintf(fpsigkn, "%14.7le %14.7le\n", (e[ik][b] - eStart) / eV, imsig[ik][b] / eV); fflush(fpsigkn);
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
			string fnamesige = "ldbd_imsige_ee_"+eep.eeMode+"_byDMD.out";
			if (exists(fnamesige)) fnamesige = "ldbd_imsige_ee_" + eep.eeMode + "_byDMD_updated.out";
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