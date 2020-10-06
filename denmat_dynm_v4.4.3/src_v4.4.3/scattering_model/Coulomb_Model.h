#pragma once
#include "common_headers.h"
#include "ScattModel_Param.h"
#include "lattice.h"
#include "parameters.h"
#include "electron.h"
#include "mymp.h"

struct coulomb_model
{
	lattice *latt;
	electron *elec;
	int nk, bStart, bEnd, nb, nv; // bStart and bEnd relative to bStart_dm
	double T, qscr2_debye, nk_full, prefac_vq, prefac_vq_bare, kF2, qTF2, fac0_Bechstedt, fac2_Bechstedt, fac4_Bechstedt;
	double **e, **f;
	std::vector<vector3<double>> qvec;
	qIndexMap *qmap;
	std::vector<complex> omega;
	std::vector<complex> vq_RPA;
	std::vector<complex> Aq_ppa, Eq2_ppa; complex v0[2];
	complex *Uih, *ovlp;

	coulomb_model(lattice *latt, parameters *param, electron *elec, int bStart, int bEnd, double dE)
		: latt(latt), elec(elec), T(param->temperature), nk(elec->nk), nk_full(elec->nk_full), 
		bStart(bStart), bEnd(bEnd), nb(bEnd - bStart), nv(elec->nv_dm - bStart), qmap(nullptr), omega(clp.nomega)
	{
		if (ionode) printf("\nInitialize screening formula %s\n", clp.scrFormula.c_str());
		if (ionode) printf("bStart = %d bEnd = %d nv = %d\n", bStart, bEnd, nv);
		if (latt->dim < 3) error_message("debye screening model for lowD is not implemented");
		prefac_vq = 4 * M_PI / clp.eps / latt->volume;
		prefac_vq_bare = 4 * M_PI / latt->volume;
		if (ionode) printf("prefac_vq = %10.3le\n", prefac_vq);
		e = trunc_alloccopy_array(elec->e_dm, nk, bStart, bEnd);
		f = trunc_alloccopy_array(elec->f_dm, nk, bStart, bEnd);
		if (clp.scrFormula == "RPA" || clp.scrFormula == "lindhard"){
			Uih = new complex[nb*elec->nb_wannier]{c0};
			ovlp = new complex[nb*nb]{c0};
		}
		if (clp.omegamax == 0) clp.omegamax = dE;
		if (clp.dynamic == "ppa") { omega.resize(2); omega[0] = c0; }

		if (clp.scrFormula == "debye" || clp.scrFormula == "Bechstedt") init_model(clp.nfreetot);
		else init_RPA();
	}
	void init(complex **dm){
		double **ft = alloc_real_array(nk, nb);
		clp.nfreetot = 0;
		for (int ik = 0; ik < nk; ik++)
		for (int b = 0; b < nb; b++){
			ft[ik][b] = real(dm[ik][b*nb + b]);
			if (b < nv) clp.nfreetot += (1 - ft[ik][b]);
			else clp.nfreetot += ft[ik][b];
		}
		clp.nfreetot /= (nk_full * latt->volume);
		if (clp.scrFormula == "debye" || clp.scrFormula == "Bechstedt") init_model(clp.nfreetot);
		else init_RPA(ft);
	}
	void init_model(double n, FILE *fp = stdout){
		if ((clp.scrFormula == "debye" || clp.scrFormula == "Bechstedt") && n <= 0)
			error_message("nfreetot must be postive");
		qscr2_debye = 4 * M_PI * n / clp.eps / T; // Eq. 11 in PRB 94, 085204 (2016)
		kF2 = std::pow(3 * M_PI*M_PI * n, 2. / 3.);
		double EF = kF2 / 2. / clp.meff;
		qTF2 = 6 * M_PI * n / clp.eps / EF;
		if (clp.scrFormula == "Bechstedt"){
			fac0_Bechstedt = 1. / (clp.eps - 1);
			fac2_Bechstedt = 1. / qTF2;
			fac4_Bechstedt = 3. / 4. / kF2 / qTF2;
		}
		if (ionode) { fprintf(fp, "#nfree = %lg cm-3, qscr2_debye = %lg (|qmin|^2 = %10.3le)\n#kF2 = %lg qTF2 = %lg\n", 
			clp.nfreetot / std::pow(bohr2cm, 3), qscr2_debye, latt->GGT.metric_length_squared(vector3<>(1. / elec->kmesh[0], 1. / elec->kmesh[1], 1. / elec->kmesh[2])), kF2, qTF2); fflush(fp); }
	}
	void init_RPA(double **ft = nullptr){
		if (clp.eppa == 0) clp.eppa = sqrt(4 * M_PI * clp.nfreetot / clp.meff);
		if (clp.dynamic == "ppa") omega[1] = ci * clp.eppa;
		if (ft != nullptr) trunc_copy_array(f, ft, nk, 0, nb);
		if (qmap == nullptr){
			qmap = new qIndexMap(elec->kmesh); qmap->build(elec->kvec, qvec);
			if (ionode && DEBUG){
				string fnameq = dir_debug + "qIndexMap.out";
				FILE *fpq = fopen(fnameq.c_str(), "w");
				fprintf(fpq, "\nPrint qIndexMap:\n"); fflush(fpq);
				for (size_t iq = 0; iq < qvec.size(); iq++){
					std::map<vector3<int>, size_t>::iterator iter = qmap->the_map.find(qmap->iqvec3(qvec[iq]));
					fprintf(fpq, "iqvec3 = (%d,%d,%d) iq = %lu\n", iter->first[0], iter->first[1], iter->first[2], iter->second);  fflush(fpq);
				}
				fclose(fpq);
			}
		}
		calc_vq_RPA();
	}
	
	complex vq(vector3<double> q, double w = 0){
		if (clp.scrFormula == "debye"){
			double q_length_square = latt->GGT.metric_length_squared(wrap(q));
			return complex(prefac_vq / (qscr2_debye + q_length_square), 0);
		}
		else if (clp.scrFormula == "Bechstedt"){
			if (q == vector3<double>(0, 0, 0)) return c0;
			double q2 = latt->GGT.metric_length_squared(wrap(q));
			double epsq = 1 + 1. / (fac0_Bechstedt + fac2_Bechstedt * q2 + fac4_Bechstedt * q2 * q2);
			return complex(prefac_vq_bare / epsq / q2, 0);
		}
		else{
			size_t iq = qmap->q2iq(q);
			if (iq == 0) return c0;
			if (clp.dynamic == "static")
				return vq_RPA[iq];
			else if (clp.dynamic == "ppa"){
				double q_length_square = latt->GGT.metric_length_squared(wrap(q));
				return (prefac_vq + Aq_ppa[iq] / (w*w - Eq2_ppa[iq])) / q_length_square; // Aq_ppa has been multiplied by prefac_vq
			}
		}
	}
	void calc_ovlp(int ik, int jk){
		hermite(elec->U[ik], Uih, elec->nb_wannier, nb);
		zgemm_interface(ovlp, Uih, elec->U[jk], nb, nb, elec->nb_wannier);
	}
	void calc_vq_RPA(){
		std::vector<std::vector<complex>> qscr2_RPA(clp.nomega, std::vector<complex>(qvec.size(), c0));
		// vq = vq0 / (1 - vq0 * sum_k [(f_k - f_k-q) / (e_k - e_k-q - w - i0)] / nk_full)
		// vq0 = e^2 / V / (eps_r * eps_0) / q^2
		// Therefore, vq = e^2 / V / (eps_r * eps_0) / (q^2 + betas^2)
		// betas^2 = - e^2 / V / (eps_r * eps_0) * sum_k [(f_k - f_k-q) / (e_k - e_k-q - w - i0)] / nk_full
		for (int iw = 0; iw < clp.nomega; iw++){
			for (int ik = 0; ik < nk; ik++)
			for (int jk = 0; jk < nk; jk++){
				vector3<double> q = elec->kvec[ik] - elec->kvec[jk];
				calc_ovlp(ik, jk);
				for (int b1 = 0; b1 < nb; b1++)
				for (int b2 = 0; b2 < nb; b2++){
					if (clp.scrFormula == "lindhard" && b1 != b2) continue;
					complex de = e[ik][b1] - e[jk][b2] - omega[iw], dfde = c0;
					if (abs(de) < 1e-8 && iw == 0 && clp.fderavitive_technique){
						double favg = 0.5 * (f[ik][b1] + f[jk][b2]);
						dfde = complex((1 - favg) * favg / T, 0); // only true for Fermi-Dirac
					}
					else dfde = complex(f[jk][b2] - f[ik][b1], 0) / (de - complex(0, clp.smearing));
					if (clp.scrFormula == "RPA") qscr2_RPA[iw][qmap->q2iq(q)] += dfde * ovlp[b1*nb + b2].norm();
					else if (clp.scrFormula == "lindhard") qscr2_RPA[iw][qmap->q2iq(q)] += dfde;
				}
			}
			axbyc(qscr2_RPA[iw].data(), nullptr, qvec.size(), 0, complex(prefac_vq / nk_full, 0), c0); // y = ax + by + c
		}
		if (clp.dynamic == "static"){
			vq_RPA.resize(qvec.size());
			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(qvec[iq]); // qvec is already wrapped to [-0.5,0.5)
				vq_RPA[iq] = complex(prefac_vq, 0) / (q_length_square + qscr2_RPA[0][iq]);
			}
		}
		else if (clp.dynamic == "ppa"){
			Aq_ppa.resize(qvec.size()); Eq2_ppa.resize(qvec.size());
			v0[0] = complex(prefac_vq, 0) / qscr2_RPA[0][0]; v0[1] = complex(prefac_vq, 0) / qscr2_RPA[1][0]; // v0[1] should be 0
			double eppa2 = clp.eppa * clp.eppa;
			for (size_t iq = 1; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(qvec[iq]); // qvec is already wrapped to [-0.5,0.5)
				complex eps0inv = c1 / (1 + qscr2_RPA[0][iq] / q_length_square);
				complex epspinv = c1 / (1 + qscr2_RPA[1][iq] / q_length_square);
				Eq2_ppa[iq] = eppa2 * (1 - epspinv) / (epspinv - eps0inv);
				Aq_ppa[iq] = prefac_vq * (1 - eps0inv) * Eq2_ppa[iq];
			}
		}

		if (ionode){
			string fnamevq = "ldbd_vq.out";
			if (exists(fnamevq)) fnamevq = "ldbd_vq_updated.out";
			FILE *fpvq = fopen(fnamevq.c_str(), "w");
			init_model(clp.nfreetot, fpvq);
			if (clp.dynamic == "ppa") { fprintf(fpvq, "wp = %lg v00 = %14.7le v0p = %14.7le\n", clp.eppa, abs(v0[0]), abs(v0[1])); fflush(fpvq); }
			fprintf(fpvq, "#|q|^2 |q_scr|^2 vq\n"); fflush(fpvq);
			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(qvec[iq]); // qvec is already wrapped to [-0.5,0.5)
				if (clp.dynamic == "static"){
					fprintf(fpvq, "%14.7le %14.7le %14.7le\n", q_length_square, abs(qscr2_RPA[0][iq]), abs(vq_RPA[iq])); fflush(fpvq);
				}
				else if (clp.dynamic == "ppa"){
					fprintf(fpvq, "%14.7le %14.7le %14.7le %14.7le %14.7le %14.7le\n", q_length_square, abs(qscr2_RPA[0][iq]), 
						abs(vq(qvec[iq], 0)), abs(vq(qvec[iq], clp.omegamax / 10)), abs(vq(qvec[iq], clp.omegamax / 2)), abs(vq(qvec[iq], clp.omegamax))); fflush(fpvq);
				}
			}
			fclose(fpvq);
		}
	}
};